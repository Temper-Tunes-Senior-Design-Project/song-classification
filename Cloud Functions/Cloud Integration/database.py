import firebase_admin
from firebase_admin import firestore

#Setup Firebase environment
app = firebase_admin.initialize_app()
db = firestore.client()

'''
Creates a database entry in the song collection under the song's UID
------------------------------------
Parameters
------------------------------------
    songUID - The unique identifier of a song provided by Spotify
    mood - The classified mood for a given song.
'''
def updateSong(songUID, mood):
    doc_ref = db.collection(u'songs').document(songUID)
    doc_ref.set({

        u'mood': mood,
    }
    )   
