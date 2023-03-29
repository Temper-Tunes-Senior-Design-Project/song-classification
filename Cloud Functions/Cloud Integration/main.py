import functions_framework
from flask import jsonify
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os

def get_variable(variable):
    return os.environ.get(variable, 'Specified environment variable is not set.')

def tempFunc():
    client = get_variable('SPOTIFY_CLIENT_ID')
    secret = get_variable('SPOTIFY_CLIENT_SECRET')
    redirect = get_variable('SPOTIFY_WEB_REDIRECT_URI')
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client,secret, redirect_uri=redirect,scope='user-library-read') )

@functions_framework.http
def generate_user_classification(request):
    spotify_access_token = request.args['spotify_token']
    userID = request.args['uid']

    print(userID, spotify_access_token)
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600'
    }
    # Jsonify predictions
    return (jsonify({}), 200, headers)