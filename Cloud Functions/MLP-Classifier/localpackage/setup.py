import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler
import requests




#need to have credentials to access API
#sp = spotipy.Spotify(auth_manager=SpotifyOAuth('86677c795a49463d9281fac012a87155','fe6f941da771447c920e02bbb2a82859', redirect_uri='http://localhost:5000',scope='user-library-read') )

def getTrackMoodsIntoDB(access_token,model,UID, user_prior_login_date=''):
    features,track_ids = getRelevantTrackFeaturesOfUnlabelledLikedSongs(access_token,UID, user_prior_login_date=user_prior_login_date)
    preprocessedFeatures = clipAndNormalize(features)
    prediction, probablity = getMoodLabelMLP(model,preprocessedFeatures)
    #either this can return these 3 pieces of information or we can save them to the database and return nothing
    return prediction, probablity, track_ids



def getRelevantTrackFeaturesOfUnlabelledLikedSongs(access_token,UID,user_prior_login_date=''):
    track_ids = retrieveTrackIds(access_token, user_prior_login_date)#supposed to be track_ids = retrieveTrackIdsToken(access_token, user_prior_login_date)
#DB track_ids = checkIdsInDatabase(track_ids)
    features = retrieveTrackFeatures(access_token, track_ids) 
#DB saveTrackFeaturesToDatabase(features)  also maybe UID to update user info
#DB updateUserLastLoginDate(UID)
    return features, track_ids

def retrieveTrackIds(sp, user_prior_login_date):
    track_ids = []
    offset = 0
    limit = 50
    liked_tracks = sp.current_user_saved_tracks(limit=limit, offset=offset)
    endLoopEarly = False
    while True:
        for item in liked_tracks['items']:
            if user_prior_login_date != '':
                if item['added_at']> user_prior_login_date:
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

def retrieveTrackIdsToken(access_token, user_prior_login_date):
    print(access_token)
    track_ids = []
    offset = 0
    limit = 50
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    url = 'https://api.spotify.com/v1/me/tracks'
    params = {'limit': limit, 'offset': offset}
    endLoopEarly = False
    while True:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            liked_tracks = response.json()
            for item in liked_tracks['items']:
                if user_prior_login_date != '':
                    if item['added_at'] > user_prior_login_date:
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
            params = {'limit': limit, 'offset': offset}
        else:
            print('Error: ', response.text)
            break
    return track_ids


def checkIdsInDatabase(track_ids):
    #check if the track_ids are already in the database
    #if they are, remove them from the track_ids list
    for track_id in track_ids:
        if track_id in database:
            track_ids.remove(track_id)
    return track_ids


def retrieveTrackFeatures(sp, track_ids):
    dfs = []
    for i in range(0, len(track_ids), 50):
        # Retrieve track features with current offset
        current_features = sp.audio_features(track_ids[i:i+50])
        
        # Convert to DataFrame
        df = pd.DataFrame(current_features)
        
        # Remove columns that we don't need
        df = df.drop(['type', 'uri', 'analysis_url', 'track_href','id'], axis=1)
        
        
        # Append to list of dataframes
        dfs.append(df)
    
    # Concatenate all dataframes into a single one
    features_df = pd.concat(dfs, ignore_index=True)
    
    features = features_df.to_numpy()
    
    return features_df

def retrieveTrackFeaturesToken(access_token, track_ids):
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    dfs = []
    for i in range(0, len(track_ids), 50):
        # Construct the URL for the current offset
        url = 'https://api.spotify.com/v1/audio-features'
        params = {'ids': ','.join(track_ids[i:i+50])}
        
        # Make the request to the API
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            # Convert the response to a DataFrame
            data = response.json()['audio_features']
            df = pd.DataFrame(data)
            
            # Remove columns that we don't need
            df = df.drop(['type', 'uri', 'analysis_url', 'track_href','id'], axis=1)
            
            # Append to list of dataframes
            dfs.append(df)
        else:
            print('Error: ', response.status_code)
    
    # Concatenate all dataframes into a single one
    features_df = pd.concat(dfs, ignore_index=True)
    
    return features_df


def saveTrackFeaturesToDatabase(features):
    #save the features to the database
    for feature in features:
        database[feature['id']] = feature
    return database

def updateUserLastLoginDate(UID):
    #update the last login date in the database
    database[UID]['last_login_date'] = datetime.now()
    return 


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
    scaler = StandardScaler()
    preprocessedFeatures = scaler.fit_transform(features)
    return preprocessedFeatures



def getMoodLabelMLP(model,songFeautures):
        prediction = model.predict(songFeautures)
        pred_probability=model.predict_proba(songFeautures)
        return prediction, pred_probability#, id

#features = getRelevantTrackFeaturesOfUnlabelledLikedSongs(sp)
#print(features)