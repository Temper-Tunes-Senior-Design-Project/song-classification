
import requests
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
from bs4 import BeautifulSoup
from bs4 import element
import re

#________________________________________________________________________________________________________________


#COMBINED MODEL OUTLINE -- 
# inputs:  sp_user( aka the client!!!), UID

# 0. get the track ids of unlabelled songs from the list of new liked songs from the user
# 1. get the song features of unlabelled songs
# 2. if possible, get the lyrics of the songs
# 3. make a prediction, based on 1 and 2
# 4. update the database and last added_at date

#outputs: verifies success or failure of the function


def get_user_song_moods_advanced(sp_user,UID):

    #0. get the track ids of unlabelled songs from the list of new liked songs from the user
    last_login = DB.get_last_login(UID)
     #IF NONE, return ''
    #If last login is a string, we need to convert it to a utc datetime object
    #if isinstance(last_login, str):
    #        last_login = datetime.strptime(last_login, '%Y-%m-%d %H:%M:%S.%f')
    
    track_ids_and_dates = retrieveTrackIdsAndDates(sp_user,last_login)

    remaining_track_ids_and_dates = DB.check_if_already_labelled(track_ids_and_dates.keys()) # drop ids of songs that are already labelled

    if len(remaining_track_ids_and_dates) == 0:
            DB.update_user_liked_songs(UID,track_ids_and_dates.keys())
            DB.update_last_added_date(UID,datetime.utcnow())
            return "success"

    #1. get the song features of unlabelled songs
    remaining_track_ids = [key for key in remaining_track_ids_and_dates.keys()]

    featuresDF = retrieveTrackFeatures(remaining_track_ids) #this should drop the ids of songs that dont have features
    featuresDict = clipAndNormalize(featuresDF)#.set_index('id').T.to_dict('list')

    #make a dictionary of song titles and artist names (for scraping lyrics)
    scraperInputs = getTitlesAndArtists(sp_user, remaining_track_ids)

    #2. if possible, get the lyrics of the songs
    for track, added_at_date in remaining_track_ids_and_dates.items():
        #2a. get lyrics
        if track in scraperInputs.keys():
            lyrics = scrapeLyrics(scraperInputs[track]['artist(s)'],scraperInputs[track]['title'])
        else:
            lyrics = []
        # 3. make a prediction, based on 1 and 2
        if track in featuresDict.keys() and len(lyrics) > 0:
            MLP_pred, MLP_pred_probability = getMoodLabelMLP([featuresDict[track]])
            BERT_pred, rely_on_linear = getOnlyMoodLabelFromLyrics(lyrics)

            if MLP_pred == BERT_pred or rely_on_linear == True:
                prediction = MLP_pred
            else:
                
                model_pred_diffs = (MLP_pred - BERT_pred)

                if MLP_pred > BERT_pred:
                    sum_probabilities = MLP_pred + model_pred_diffs
                else:
                    sum_probabilities = MLP_pred - model_pred_diffs
                #if sum_probabilities outside of below 0, then do 8-sum_probabilities
                if sum_probabilities < 0:
                    prediction = 8 + sum_probabilities
                elif sum_probabilities > 7:
                    prediction = sum_probabilities - 7
                else:
                    prediction = sum_probabilities
            #4. update database
            DB.add_song_mood(track,prediction)
            DB.update_user_liked_songs(UID,track)
            DB.update_last_added_date(UID,added_at_date)
        elif len(lyrics) > 0: #this case might never occur
            BERT_pred, rely_on_linear = getOnlyMoodLabelFromLyrics(lyrics)
            if rely_on_linear == False:
                DB.add_song_mood(track,BERT_pred)
                DB.update_user_liked_songs(UID,track)
                DB.update_last_added_date(UID,added_at_date)
        elif track in featuresDict.keys():
            MLP_pred, MLP_pred_probability = getMoodLabelMLP([featuresDict[track]])
            DB.add_song_mood(track,MLP_pred)
            DB.update_user_liked_songs(UID,track)
            DB.update_last_added_date(UID,added_at_date)
        else:
            print("no features or lyrics available for this song!")
                #go to next iteration of the loop
                #maybe want to return a warning
            continue

    return "success"

#________________________________________________________________________________________________________________
### Helper Functions of get_User_Song_Moods_2_models
#________________________________________________________________________________________________________________

def retrieveTrackIdsAndDates(sp, user_prior_login_date):
    track_ids = {}
    offset = 0
    limit = 50
    liked_tracks = sp.current_user_saved_tracks(limit=limit, offset=offset)
    endLoopEarly = False
    while True:
        for item in liked_tracks['items']:
            song_added_at = datetime.strptime(item['added_at'],'%Y-%m-%dT%H:%M:%SZ')
            if user_prior_login_date != '':
                if song_added_at > user_prior_login_date:
                    track_ids[item['track']['id']]= song_added_at
                else:
                    endLoopEarly = True
                    break
            else:
                track_ids[item['track']['id']]= song_added_at
        offset += limit
        
        if len(liked_tracks['items']) < limit or endLoopEarly:
            # All tracks have been retrieved
            break
        
        liked_tracks = sp.current_user_saved_tracks(limit=limit, offset=offset)
    
    return track_ids


def retrieveTrackFeatures(sp, track_ids):
    dfs = []

    for i in range(0, len(track_ids), 50):
        # Retrieve track features with current offset
        current_features = sp.audio_features(track_ids[i:i+50])
        
        # Convert to DataFrame
        df = pd.DataFrame(current_features)
        
        # Remove columns that we don't need
        df = df.drop(['type', 'uri', 'analysis_url', 'track_href'], axis=1)
        
        
        # Append to list of dataframes
        dfs.append(df)
    
    # Concatenate all dataframes into a single one
    features_df = pd.concat(dfs, ignore_index=True)
    
    return features_df

def clipAndNormalize(features):
    #clip the features to the range of the training data
    features['danceability'] = features['danceability'].clip(lower=0.22718080000000002, upper=0.906)
    features['energy'] = features['energy'].clip(lower=0.03545904, upper=0.978)
    features['loudness'] = features['loudness'].clip(lower=-26.4981552, upper=-1.6015904000000007)
    features['speechiness'] = features['speechiness'].clip(lower=0.0257, upper=0.46640959999999926)
    features['acousticness'] = features['acousticness'].clip(lower=8.353136000000001e-05, upper=0.9884095999999992)
    features['instrumentalness'] = features['instrumentalness'].clip(lower=0.0, upper=0.956)
    features['liveness'] = features['liveness'].clip(lower=0.0494, upper=0.697)
    features['valence'] = features['valence'].clip(lower=0.0382, upper=0.923)
    features['tempo'] = features['tempo'].clip(lower=63.7631808, upper=188.00344319999996)
    features['duration_ms'] = features['duration_ms'].clip(lower=88264.8768, upper=372339.1727999991)
    features['time_signature'] = features['time_signature'].clip(lower=3.0, upper=5.0)
    
    #normalize the data
    scaler = pickle.load(open('scaler2.pkl', 'rb'))
    #fit on all columns except the track id
    rawfeatures = features.drop(['id'], axis=1)
    preprocessedFeatures = scaler.transform(rawfeatures)

    #convert to dictionary, with track id as key
    preprocessedFeatures = pd.DataFrame(preprocessedFeatures, columns=rawfeatures.columns)
    preprocessedFeatures['id']= features['id']
    preprocessedFeatures = preprocessedFeatures.set_index('id').T.to_dict('list')
    return preprocessedFeatures


def getTitlesAndArtists(sp, track_ids):
    titleArtistPairs = {}
    for i in range(0,len(track_ids),50):
        tracks = sp.tracks(track_ids[i:i+50])
        for track in tracks['tracks']:
            title=track['name']
            #check if the track ends with (feat. artist) using a regex
            if re.search(r' \(feat. .*\)$', title):
                #remove the (feat. artist) from the title
                title = re.sub(r' \(feat. .*\)$', '', title)

            artists=[]
            for artist in track['artists']:
                artists.append(artist['name'])
            titleArtistPairs[track['id']] = {'title':title,'artist(s)':artists}

    return titleArtistPairs

def getScrapedLyrics(scraperInputs):
        all_lyrics_dict = {}
        for id, songInfo in scraperInputs.items():
                #maybe add a sleep or something to prevent getting blocked
                lyrics = scrapeLyrics(songInfo['artist(s)'],songInfo['title'])
                if len(lyrics) > 0:
                        all_lyrics_dict[id]=lyrics
        return all_lyrics_dict


def getMoodLabelMLP(songFeautures):
        with open('MLP2.pkl','rb') as f:
            model = pickle.load(f)
        prediction = model.predict(songFeautures)
        pred_probability=model.predict_proba(songFeautures)
        return prediction, pred_probability
