
#add framework and other imports

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
