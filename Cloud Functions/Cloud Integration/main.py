import functions_framework
from flask import jsonify
import spotipy
import classifier
from spotipy.oauth2 import SpotifyOAuth
import os

'''
Gets enviroment variables securely stored in .env.yaml
------------------------------------
Parameters
------------------------------------
   variable - The variable name to access
'''
def get_variable(variable):
    return os.environ.get(variable, 'Specified environment variable is not set.')

'''
Authenticates the user with the spotipy client
------------------------------------
Parameters
------------------------------------
   token - A spotify OAuth2.0 token
'''
def loadSpotipyClient(token):
    client = get_variable('SPOTIFY_CLIENT_ID')
    secret = get_variable('SPOTIFY_CLIENT_SECRET')
    redirect = get_variable('SPOTIFY_WEB_REDIRECT_URI')
    #sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client,secret, redirect_uri=redirect,scope='user-library-read') )
    return spotipy.Spotify(auth=token)

'''
Perform classification on all of the songs within a users liked song library
------------------------------------
Parameters
------------------------------------
   request - A flask request object containing the parameters passed to the server
'''
@functions_framework.http
def generate_user_classification(request):
    spotify_access_token = request.args['spotify_token']
    userID = request.args['uid']
    client = loadSpotipyClient(spotify_access_token);

    print(userID, spotify_access_token)

    prediction, probablity, track_ids= classifier.populateTrackMoods(spotify_access_token,userID,model)
    pred_count = len(prediction)
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600'
    }
    # Jsonify predictions
    return (jsonify({}), 200, headers)