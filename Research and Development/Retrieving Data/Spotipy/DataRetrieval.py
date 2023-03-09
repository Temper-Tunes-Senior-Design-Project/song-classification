import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd


#need to have credentials to access API
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials('bf1ba68423404778a60bcf3dee58d199','7365dc611a2d4ddba4ad61343f0b64d7'))

#NOTE: playlistID is a string, preferably uri/url over href; HARD CODED PERSONAL PATH EXPORTING CSV and I expect CSVFileName to end in .csv
def getSongMetadataFromPlaylist(playlistID, affect,CSVFileName='x', dataSubFolderLocation = 'emotions'): 
    playlistBatches = getAllTracksFromPlaylist(playlistID)
    playlistName = sp.playlist(playlistID)['name']

    allSongFeatures = []
    
    for playlistBatch in playlistBatches:
        
        allSongFeatures.extend(getSongData(playlistBatch, playlistName, affect))
        
    df = pd.DataFrame(allSongFeatures)

    CSVFilePath = exportPathToCSV(playlistName,  CSVFileName, dataSubFolderLocation,affect)

    df.to_csv(CSVFilePath, index=False)
    
    print('Warning: I did not add a check to see if .csv file alread exists so it will overwrite the file if it already exists!')
    print('Extracted', len(df), 'songs metadata from', playlistName,'.')
    return df


def getAllTracksFromPlaylist(playlistID):
    tracks = []
    offset = 0
    limit = 100
    
    while True:
        playlist = sp.playlist_tracks(playlistID, offset=offset, limit=limit)['items']
        tracks.append(playlist)
        
        if len(playlist) < limit:
            break
            
        offset += limit
        print("obtained batch", offset//limit, "of songs")
    return tracks


def getSongData(playlist, playlistName,  affect ):
    songDataExtractedCount=0
    song_uris = []
    valid_song_indexes = []
    for i, song in enumerate(playlist):
        if song['track'] is not None:
            song_uris.append(song['track']['uri'])
            valid_song_indexes.append(i)
    song_features = sp.audio_features(song_uris)
    song_features_with_metadata_batch = []
    for i, feature in enumerate(song_features):
        if feature is not None:
            song_index = valid_song_indexes[i]
            songName = playlist[song_index]['track']['name']
            song_features_with_metadata_batch.append(feature)
            song_features_with_metadata_batch[-1]['song'] = songName
            song_features_with_metadata_batch[-1]['playlist'] = playlistName
            song_features_with_metadata_batch[-1]['affect'] = affect
            songDataExtractedCount += 1
    print(f"{songDataExtractedCount} of {len(playlist)} in batch extracted")
    
    return song_features_with_metadata_batch


def exportPathToCSV(playlistName,  CSVFileName, dataSubFolderLocation,affect):
    if dataSubFolderLocation != '' and dataSubFolderLocation[-1] !='/':
        dataSubFolderLocation += f'/{affect}/'
        #this assumes affect is supposed to be part of subfolder name
    elif dataSubFolderLocation =='':
        dataSubFolderLocation = f'{affect}/'
    
    if CSVFileName == 'x':
        CSVFilePath = 'C:/Users/mlar5/OneDrive/Desktop/Code Folder/Python Projects/IRL projects/Aspire - Affective Computing Project/Playlists Data/Audio Analysis/' + dataSubFolderLocation+ playlistName
    else:
        CSVFilePath = 'C:/Users/mlar5/OneDrive/Desktop/Code Folder/Python Projects/IRL projects/Aspire - Affective Computing Project/Playlists Data/Audio Analysis/'+ dataSubFolderLocation + CSVFileName

    if CSVFilePath[-4:] != '.csv':
        CSVFilePath += '.csv'
    return CSVFilePath


############################################      "MAIN"      #######################################################


playlistID = 'https://open.spotify.com/playlist/0uiT9gi9uIrbtuj7NGHsYb?si=a743b183e4f84d06'
#CSVFile = 'Rock Testing Concat.csv'

affect = ['sad','anxious','energetic','excited','happy','calm','relaxed','depressed']
#getSongMetadataFromPlaylist(playlistID, affect[1], CSVFile, usingAudioAnalysis=True)
#dont have to use custom csv filename, as shown below!

getSongMetadataFromPlaylist(playlistID, affect[7])
#print the current directory