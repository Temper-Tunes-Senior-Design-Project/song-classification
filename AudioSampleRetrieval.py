import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

#need to have credentials to access API
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials('bf1ba68423404778a60bcf3dee58d199','7365dc611a2d4ddba4ad61343f0b64d7'))



def addAudioSampleURLsToDF(src, exportToCSV = False, dest = 'WithAudioURLs.csv' ):
    originalDF = pd.read_csv(src)
    previewURLs = []
    audioSampleDF = originalDF.copy()

    for songID in originalDF['uri']:      
        currTrack = sp.track(songID)
        previewURLs.append(currTrack['preview_url'])
    audioSampleDF['preview_url'] = previewURLs

    if exportToCSV:
        if dest == 'WithAudioURLs.csv':
            updatedDestination = src[:-4]+dest
            audioSampleDF.to_csv(updatedDestination, index=False)
        else:    
            audioSampleDF.to_csv(dest, index=False)
    return audioSampleDF



dfWithAudioLinks = addAudioSampleURLsToDF('firstOutlierFree.csv', exportToCSV = True, dest ='outlierFreeWithAudioLinks.csv')
