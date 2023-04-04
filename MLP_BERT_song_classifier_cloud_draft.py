
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle
#this folder location will be changed to DB location in actual cloud function
from os import chdir
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from bs4 import BeautifulSoup
from bs4 import element
from copy import deepcopy
import re

#________________________________________________________________________________________________________________


#COMBINED MODEL OUTLINE -- 
# inputs:  sp_user( aka the client!!!), UID
#^note: we could pass in the user's last login date, instead of UID if we waned to, 
# as this would provide beter separation of concerns of updating last login date vs the function 
# thats supposed to be getting the song moods

#outputs: verifies success or failure of the function

from datetime import datetime

def get_user_song_moods_advanced(sp_user,UID):

        #0. get the track ids of unlabelled songs from the list of new liked songs from the user
        last_login = DB.get_last_login(UID) #IF NONE, return ''

        #If last login is a string, we need to convert it to a utc datetime object
        #if isinstance(last_login, str):
        #        last_login = datetime.strptime(last_login, '%Y-%m-%d %H:%M:%S.%f')
        
        track_ids = retrieveTrackIds(sp_user,last_login)

        remaining_track_ids = DB.check_if_already_labelled(track_ids)

        if len(remaining_track_ids) == 0:
                DB.update_user_liked_songs(UID,track_ids)
                DB.update_last_login(UID,datetime.utcnow())
                return "success"

        #1. get the song features of unlabelled songs

        featuresDF = retrieveTrackFeatures(remaining_track_ids) #this should drop the ids of songs that dont have features
        featuresDict = clipAndNormalize(featuresDF)#.set_index('id').T.to_dict('list')

        #2. if possible, get the lyrics of the songs

        #make a dictionary of song titles and artist names
        scraperInputs = getTitlesAndArtists(sp_user, remaining_track_ids)


        all_lyrics_dict = getScrapedLyrics(scraperInputs)

        #MAYBE Tokenize THE LYRICS HERE (use batch processing?!) 
        #would need to verify compute restrictions of cloud function first!

        overlap_keys = [key for key in featuresDict.keys() if key in all_lyrics_dict.keys()]
        only_features = [key for key in featuresDict.keys() if key not in overlap_keys]
        only_lyrics = [key for key in all_lyrics_dict.keys() if key not in overlap_keys]


        #3 get predictions and update database

        #3a. if there are no features or lyrics, return an error
        if len(featuresDict.keys()) == 0 and len(all_lyrics_dict.keys()) == 0:
                #probably just return a flag that says there are no features or lyrics not the response here
                return "songs found but no data available for predictions"
        
        else:
                predictions = {}
                chdir('C:/Users/mlar5/OneDrive/Desktop/Code Folder/198 Senior Design/Models/Spotipy')
                #pickle in MLP model
                MLP_model = pickle.load(open('MLP1.pkl', 'rb'))

                #load in local copy of BERT model
                BERT_model = AutoModelForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")
                BERT_Tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")

                #for first version, tokenize the lyrics and then pass then to the model inside the for loop

                for key in overlap_keys.keys(): #- could probably be done in batches regardless
                        MLP_pred, MLP_pred_probability = getMoodLabelMLP(MLP_model,featuresDict[key])

                        #BERT_pred, BERT_pred_probability = getMoodLabelBERT(BERT_model,all_lyrics_dict[key])
                        BERT_pred, MLP_flag = getOnlyMoodLabelFromLyrics(all_lyrics_dict[key])
                        if MLP_pred == BERT_pred:
                                prediction = MLP_pred
                        else:
                                #add probabilities and choose the label with the highest probability
                                sum_probabilities = MLP_pred_probability + BERT_pred_probability
                                prediction = np.argmax(sum_probabilities)
                        predictions[key]=prediction

                for key in only_features:
                        MLP_pred, MLP_pred_probability = getMoodLabelMLP(MLP_model,featuresDict[key])
                        predictions[key]=MLP_pred

                for key in only_lyrics:
                        #BERT_pred, BERT_pred_probability = getMoodLabelBERT(BERT_model,all_lyrics_dict[key])
                        BERT_pred, MLP_flag = getOnlyMoodLabelFromLyrics(all_lyrics_dict[key])
                        predictions[key]=BERT_pred

                DB.add_song_moods(predictions)
                
                DB.update_user_liked_songs(UID,predictions.keys()) 
                # ^^Need to add a check to discard songs no longer on spotify, 
                # otherwise we might recommend songs that are no longer on spotify

                DB.update_last_login(UID,datetime.utcnow())

        return "success"

#________________________________________________________________________________________________________________
### Helper Functions of get_User_Song_Moods_2_models
#________________________________________________________________________________________________________________

def retrieveTrackIds(sp, user_prior_login_date):
    track_ids = []
    offset = 0
    limit = 50
    liked_tracks = sp.current_user_saved_tracks(limit=limit, offset=offset)
    endLoopEarly = False
    while True:
        for item in liked_tracks['items']:
            if user_prior_login_date != '':
                if datetime.strptime(item['added_at'],'%Y-%m-%dT%H:%M:%SZ')> user_prior_login_date:
                    track_ids.append(item['track']['id'])
                else:
                    endLoopEarly = True
                    break
            else:
                track_ids.append(item['track']['id'])
        offset += limit
        
        if len(liked_tracks['items']) < limit or endLoopEarly:
            # All tracks have been retrieved
            break
        
        liked_tracks = sp.current_user_saved_tracks(limit=limit, offset=offset)
    
    return track_ids


'''def getSongFeaturesToDict(track_ids):
        featuresDict = {}
        for i in range(0,len(track_ids),100):
                features = sp.audio_features(track_ids[i:i+100])
                for feature in features:
                        if feature != None:
                                featuresDict[feature['id']]=feature
        return featuresDict'''

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
    
    
    #convert to dictionary, with track id as key
    #featuresDict = features_df.set_index('id').T.to_dict('list')
    
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
    scaler = pickle.load(open('scaler1.pkl', 'rb'))
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





#________________________________________________________________________________________________________________
#                  SCRPAER CODE
#________________________________________________________________________________________________________________

#Helps parse miscellaneous tags <i>, </br>, etc,.
def _lyricsHelper(html, lyrics_list):
    for tag in html.childGenerator():
        if type(tag) == element.NavigableString:
            _handleLyricAppend(lyrics_list, tag.text.strip())
        elif tag.name == 'br' and lyrics_list[len(lyrics_list) - 1] != '':
            lyrics_list.append('')
        elif html.name == 'a':
            _lyricsHelper(tag, lyrics_list)

#Reads the HTML for lyrics dividers (if they exist) and appends the lyrics line by line to a list
def _getLyricsFromHTML(html):
    lyrics = html.findAll("div", {"data-lyrics-container" : "true"})
    lyrics_list = ['']
    for segment in lyrics:
        for a in segment.childGenerator():
            lyric = None
            if type(a) == element.NavigableString:
                lyric = a.strip()
                _handleLyricAppend(lyrics_list, lyric)
            else:
                _lyricsHelper(a, lyrics_list)
            if a.name == 'br' and lyrics_list[len(lyrics_list) - 1] != '':
                lyrics_list.append('')
    return lyrics_list

#Helper function to handle appending and manipulating lyrics_list. A new line is generated only for </br> tags
def _handleLyricAppend(lyrics_list, lyric):
    if lyric is not None:
        last_index = len(lyrics_list) - 1
        #Handle special cases (parenthesis and symbols stick with words for instance)
        if lyrics_list[last_index] != '' and (lyrics_list[last_index][-1] in ['(','[','{','<'] or lyric in [')',']','}','>','!','?',',','.']):
            lyrics_list[last_index] += lyric
        else:
            lyrics_list[last_index] += " " + lyric if lyrics_list[last_index] != '' else lyric

#Determines whether a song will need to be translated (returns the link if it does, otherwise returns None)
def _getSongTranslationLink(html):
    translation_tags = html.findAll('a', class_='TextButton-sc-192nsqv-0 hVAZmF LyricsControls__TextButton-sghmdv-6 hXTHjZ')
    for tag in translation_tags:
        if "english-translations" in tag['href']:
            return tag['href']
    return None

#Determines whether a page exists
def _pageExists(html):
    return html.find('div', class_='render_404') == None
        
#function to scrape lyrics from genius, takes an array of artists, and songname
def scrapeLyrics(artistnames, songname):
    lyrics_list = []
    found = False
    i = 0
    html = None
    while i < len(artistnames) and not(found):
        artistname = artistnames[i]
        artistname2 = str(artistname.replace(' ','-')) if ' ' in artistname else str(artistname)
        songname2 = str(songname.replace(' ','-')) if ' ' in songname else str(songname)
        page_url = 'https://genius.com/'+ artistname2 + '-' + songname2 + '-' + 'lyrics'
        page = requests.get(page_url)
        html = BeautifulSoup(page.text, 'html.parser') 
        found = _pageExists(html)
        i += 1
    if found:
        #check if there is an english translation
        translation_url = _getSongTranslationLink(html)
        if translation_url is not None:
            page = requests.get(translation_url)
            html = BeautifulSoup(page.text, 'html.parser') 
            lyrics_list = _getLyricsFromHTML(html)
        else:
            #If there isn't a translation, make sure it's in english in the first place
            english = False
            for script in html.findAll('script'):
                if "language\\\":\\\"en" in str(script):
                    english = True
            if english: lyrics_list = _getLyricsFromHTML(html)
    return lyrics_list
    


#________________________________________________________________________________________________________________
#                  MAIN
#________________________________________________________________________________________________________________

def generate_song_classification_advanced(request):
        spotify_access_token = request.args['spotify_token']
        userID = request.args['uid']
        client = loadSpotipyClient(spotify_access_token)

        feedback = get_user_song_moods_advanced(client, userID)


        #CORS-Policy Headers
        headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600'
        }
        # Jsonify predictions
        if feedback == "success":
                return (jsonify({}), 200, headers)
        else:
                return (jsonify({"warning":feedback}), 200, headers)
