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
    #client = '91654452e7694f638af81a18c8dedf2f'
    #ecret = '89210f07be76488b90792983d9fe99a7'
    #redirect = 'https://mood-swing-spotify-redirect.web.app'
    print(client, secret, redirect, sep = "\n")
    scopes = "app-remote-control,user-modify-playback-state,playlist-read-private,user-library-read"
    auth_manager = SpotifyOAuth(client, secret, redirect, scope=scopes)
    return spotipy.Spotify(auth=token, auth_manager=auth_manager)


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
    client = loadSpotipyClient(spotify_access_token)

    prediction, probablity, track_ids = classifier.populateTrackMoods(client, userID)
    #CORS-Policy Headers
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600'
    }
    # Jsonify predictions
    return (jsonify({}), 200, headers)
