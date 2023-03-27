## NOTE: This should assume that the data is already in a given format from SPOTPY and scaled

from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
from flask import jsonify
from google.cloud import storage

def getMoodLabelMLP(request):
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

    if request_json and "SPD" in request_json: 
        # Load all files from storage bucket storage and instantiate
        storage_client = storage.Client()

        bucket = storage_client.bucket('software-engineering-7af33.appspot.com')

        #setattr(sys.modules["__main__"],'NCF',NCF)
        blob = bucket.blob('MLP1.pkl')
        pickle_in = blob.download_as_string()
        model = pickle.loads(pickle_in)


        #make a test example of a numpy array with the correct shape and import numpy
        testExample = np.array([-1.30196675,  0.80096565,  0.2161288 ,  0.62748397,  0.8056064 ,
            -0.15219274, -0.98604416, -0.54784952, -0.3229561 , -1.13850136,
                0.1627538 ,  1.00204954,  0.18734159]).reshape(1, -1)
        prediction = model.predict(testExample)
        pred_probability=model.predict_proba(testExample)

        #return the label and the probability of the label
        return jsonify({"label":prediction, "probability":pred_probability}),200, headers

        #return prediction, pred_probability

"""
        recommendations=''
     #recommendations = getUserRecommendationsShort(usersMovies,model,users,moviePool, links,movieTitles, usersCorrespondingClusters, medoidUsers)
        
        if recommendations == "tt0000000":
            return jsonify({"problem":"No user movies match the models movies"}),200,headers
        return jsonify({"movies":recommendations}),200, headers
    
    return jsonify({"problem":"No movies were received to make a recommendation"}), 200, headers
"""