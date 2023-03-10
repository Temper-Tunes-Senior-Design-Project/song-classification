import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd


#need to have credentials to access API
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials('bf1ba68423404778a60bcf3dee58d199','7365dc611a2d4ddba4ad61343f0b64d7'))

#NOTE: playlistID is a string, preferably uri/url over href; HARD CODED PERSONAL PATH EXPORTING CSV and I expect CSVFileName to end in .csv
def getSongMetadataFromPlaylist(playlistID, affect,genre,CSVFileName='x', dataSubFolderLocation = 'emotions',showWarning=True): 
    playlistBatches = getAllTracksFromPlaylist(playlistID)
    playlistName = sp.playlist(playlistID)['name']

    allSongFeatures = []
    
    for playlistBatch in playlistBatches:
        
        allSongFeatures.extend(getSongData(playlistBatch, playlistName, affect,genre))

        
    df = pd.DataFrame(allSongFeatures)

    CSVFilePath = exportPathToCSV(playlistName,  CSVFileName, dataSubFolderLocation,affect)

    df.to_csv(CSVFilePath, index=False)
    
    if showWarning:
        print('Warning: I did not add a check to see if .csv file alread exists so it will overwrite the file if it already exists!')
    print(f'Extracted {len(df)} songs metadata from {playlistName} as {affect}, {genre}.')
    return df


def getAllTracksFromPlaylist(playlistID):
    tracks = []
    offset = 0
    limit = 100
    
    while True:
        playlist = sp.playlist_tracks(playlistID, offset=offset, limit=limit)['items']
        tracks.append(playlist)
        print("obtained batch", (offset//limit)+1, "of songs")

        if len(playlist) < limit:
            break
            
        offset += limit
    return tracks


def getSongData(playlistBatch, playlistName,  affect,genre ):
    songDataExtractedCount=0
    song_uris = []
    valid_song_indexes = []
    for i, song in enumerate(playlistBatch):
        if song['track'] is not None:
            song_uris.append(song['track']['uri'])
            valid_song_indexes.append(i)
    song_features = sp.audio_features(song_uris)
    song_features_with_metadata_batch = []
    for i, feature in enumerate(song_features):
        if feature is not None:
            song_index = valid_song_indexes[i]
            songName = playlistBatch[song_index]['track']['name']
            song_features_with_metadata_batch.append(feature)
            song_features_with_metadata_batch[-1]['song'] = songName
            song_features_with_metadata_batch[-1]['playlist'] = playlistName
            song_features_with_metadata_batch[-1]['affect'] = affect
            song_features_with_metadata_batch[-1]['genre'] = genre
            songDataExtractedCount += 1
    print(f"{songDataExtractedCount} of {len(playlistBatch)} in batch extracted")
    
    return song_features_with_metadata_batch


def exportPathToCSV(playlistName,  CSVFileName, dataSubFolderLocation,affect):
    if dataSubFolderLocation != '' and dataSubFolderLocation[-1] !='/':
        dataSubFolderLocation += f'/{affect}/'
        #this assumes affect is supposed to be part of subfolder name
    elif dataSubFolderLocation =='':
        dataSubFolderLocation = f'{affect}/'
    
    if CSVFileName == 'x':
        playlistName = playlistName.replace('/',' and ')
        playlistName = playlistName.replace(':',' - ')
        playlistName = playlistName.replace('?','')
        playlistName = playlistName.replace('!','')
        playlistName = playlistName.replace('(','')
        playlistName = playlistName.replace(')','')
        
        CSVFilePath = 'C:/Users/mlar5/OneDrive/Desktop/Code Folder/Python Projects/IRL projects/Aspire - Affective Computing Project/Playlists Data/Audio Features/' + dataSubFolderLocation+ playlistName
    else:
        #check if CSVFileName has a \ in it, if so, then change it to not be treated as an escape character
        CSVFilePath = 'C:/Users/mlar5/OneDrive/Desktop/Code Folder/Python Projects/IRL projects/Aspire - Affective Computing Project/Playlists Data/Audio Features/'+ dataSubFolderLocation + CSVFileName

    if CSVFilePath[-4:] != '.csv':
        CSVFilePath += '.csv'
    return CSVFilePath


############################################      "MAIN"      #######################################################


playlistID = """
https://open.spotify.com/playlist/2gxwsEcFCccHyTS2NrQA7U?si=4d0b8e0979e344cf
"""
#CSVFile = 'Rock Testing Concat.csv'

affect = ['sad','angry','energetic','excited','happy','content','calm','depressed']
#           0      1           2          3        4      5chill?    6         7

#getSongMetadataFromPlaylist(playlistID, affect[1], CSVFile, usingAudioAnalysis=True)
#dont have to use custom csv filename, as shown below!
genre = ['rock','pop','metal','rap','country','latino','instrumental','EDM','R&B','Kpop']
#           0      1     2       3      4       5          6            7     8     9

getSongMetadataFromPlaylist(playlistID.splitlines()[1], affect[1], genre[1],showWarning=False)
#NOTE: Pop is a generally referring to playlists with a mix of other genres