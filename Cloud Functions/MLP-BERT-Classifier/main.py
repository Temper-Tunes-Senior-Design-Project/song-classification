from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
from flask import jsonify
from google.cloud import storage
from localpackage import getMoodLabelMLP


#input: DO we get a list of song UIDs or one song UID per request?
#also do we get the song features or do we get them from the database?
#like do we have a cloud function putting the song features in the database and then we get them from the database?
#main confusion is how to access data from user's spotify account? Does the user have to give us access to their spotify account within the cloud function?
def getMoodLabel(request):
    # CORS POLICY
    if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',#'http://127.0.0.1:5500'
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            #'Access-Control-Allow-Credentials': 'true',

            'Access-Control-Max-Age': '3600'
        }

        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }


    # Get the request body and call a function to return label in the response
    #from flask import jsonify

    
    request_json = request.get_json()

    if request_json and "UID" in request_json: 
        # Load all files from storage bucket storage and instantiate
        storage_client = storage.Client()

        bucket = storage_client.bucket('software-engineering-7af33.appspot.com')

        #setattr(sys.modules["__main__"],'NCF',NCF)
        blob = bucket.blob('MLP1.pkl')
        pickle_in = blob.download_as_string()
        model = pickle.loads(pickle_in)


        #1. if possible and not already done, get the song features from the database using the song UID from the request
        # features = DB.getSongFeatures(request_json['UID']) OR spotify.getSongFeatures(request_json['UID'])

        #1.5 save the song ids to the user's database


        #2. if possible, get the lyrics from the request
        #lyrics = Uriyafunction.getLyrics(request_json['UID'])

        #3. determine what model to use or use both!

        #if lyrics not None and features not None:
            #MLP_pred, MLP_pred_probability = getMoodLabelMLP(model,features)

            #BERT_pred, BERT_pred_probability = getMoodLabelBERT(model,lyrics)

            #if MLP_pred == BERT_pred:
                #prediction = MLP_pred
            #else:
                #add probabilities and choose the label with the highest probability
                #sum_probabilities = MLP_pred_probability + BERT_pred_probability
                #prediction = np.argmax(sum_probabilities)

        #elif lyrics not None:
            #BERT_pred, BERT_pred_probability = getMoodLabelBERT(model,request_json)
            #prediction = BERT_pred
        #elif features not None:
            #MLP_pred, MLP_pred_probability = getMoodLabelMLP(model,request_json)
            #prediction = MLP_pred
        #else:
            #prediction = "No features or lyrics were found for this song"
            #return jsonify({"error":"No UID was provided"}),400, headers

        #return the label and the probability of the label
        return jsonify({"label":prediction}),200, headers
    else:
        return jsonify({"error":"No UID was provided"}),400, headers