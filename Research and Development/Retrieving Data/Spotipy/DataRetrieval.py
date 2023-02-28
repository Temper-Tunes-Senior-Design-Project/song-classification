import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import warnings
#ignoring the warning that the DataFrame.append() method is deprecated and will eventually be removed.
warnings.filterwarnings(action='ignore', category=FutureWarning)


#need to have credentials to access API
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials('bf1ba68423404778a60bcf3dee58d199','7365dc611a2d4ddba4ad61343f0b64d7'))

#NOTE: playlistID is a string, preferably uri/url over href; HARD CODED PERSONAL PATH EXPORTING CSV and I expect CSVFileName to end in .csv
def getSongMetadataFromPlaylist(playlistID, affect,CSVFileName='x',usingAudioAnalysis=False, dataSubFolderLocation = ''): 
    playlist = sp.playlist_tracks(playlistID)['items']
    playlistName = sp.playlist(playlistID)['name']

    df = makeAudioOrFeatureDF(usingAudioAnalysis) 
    moreSongsToLoad = True
    currOffset = 0
    songDataExtractedCount =0
    killCount = 0
    
    while moreSongsToLoad:
        
        df, playlist, songDataExtractedCount = getSongData(playlist, playlistName, df,affect,usingAudioAnalysis, songDataExtractedCount)

        moreSongsToLoad, playlist, currOffset = checkForMoreSongs(playlist,playlistID, currOffset)

        moreSongsToLoad, killCount = infLoopCheck(killCount, moreSongsToLoad)
        
    exportToCSV(playlistName, usingAudioAnalysis, df, songDataExtractedCount, CSVFileName, dataSubFolderLocation)
    

#NOTE: this is directly relying on the dictionaries being consistent
#no error/exception handling being done here
def getSongData( playlist, playlistName,df,affect,usingAudioAnalysis, songDataExtractedCount=0):
    for song in playlist:
        songName = song['track']['name']

        songURI=song['track']['uri']

        if usingAudioAnalysis:
            # (currently unsure how its breaking it down in excel file!!!)
            featuresOrAnalysis = sp.audio_analysis(songURI)
            print(featuresOrAnalysis)
        else:
            featuresOrAnalysis = sp.audio_features(songURI)[0]

        featuresOrAnalysis['song'] = songName
        featuresOrAnalysis['playlist']= playlistName
        featuresOrAnalysis['affect'] = affect

        #seriesToJoin = pd.Series(featuresOrAnalysis)
        #df = pd.concat([df,seriesToJoin], ignore_index=True)
        #NOTE: df.append is going to be removed in the future, but i cant currently get it to work the preferred way using pd.concat

        df = df.append(featuresOrAnalysis, ignore_index=True)


        songDataExtractedCount +=1
        print(songDataExtractedCount)
        if usingAudioAnalysis:
            print('Currently I have a break line to only retreive the audio analysis of the first song in a playlist since I havent used it much')
            break
    return df, playlist, songDataExtractedCount

#NOTE: the songs of a playlist r broken up into batches of 100
def checkForMoreSongs(playlist,playlistID, currOffset):
    if len(playlist) < 100 and len(playlist)>-1 :
        return False, playlist, currOffset
    elif len(playlist) == 100:
        currOffset +=100
        playlist = sp.playlist_tracks(playlistID, offset=currOffset)['items']
        return True, playlist, currOffset
    else:
        print("Problem determining playlist size!")
        return False, playlist, currOffset

def exportToCSV(playlistName, usingAudioAnalysis, df, songDataExtractedCount, CSVFileName, dataSubFolderLocation):
    if dataSubFolderLocation != '' and dataSubFolderLocation[-1] !='/':
        dataSubFolderLocation += '/'
    
    if usingAudioAnalysis:
        if CSVFileName == 'x':
            CSVFilePath = 'C:/Users/mlar5/OneDrive/Desktop/Code Folder/Pytorch Projects/IRL projects/Aspire - Affective Computing Project/Playlists Data/Audio Analysis/' + dataSubFolderLocation+ playlistName + '.csv'
        else:
            CSVFilePath = 'C:/Users/mlar5/OneDrive/Desktop/Code Folder/Pytorch Projects/IRL projects/Aspire - Affective Computing Project/Playlists Data/Audio Analysis/'+ dataSubFolderLocation + CSVFileName
    elif CSVFileName == 'x':
        CSVFilePath = 'C:/Users/mlar5/OneDrive/Desktop/Code Folder/Pytorch Projects/IRL projects/Aspire - Affective Computing Project/Playlists Data/Audio Features/' + dataSubFolderLocation + playlistName + '.csv'
    else:
        CSVFilePath = 'C:/Users/mlar5/OneDrive/Desktop/Code Folder/Pytorch Projects/IRL projects/Aspire - Affective Computing Project/Playlists Data/Audio Features/' + dataSubFolderLocation + CSVFileName

    df.to_csv(CSVFilePath, index=False)
    print('Warning: I did not add a check to see if .csv file alread exists so it will overwrite the file if it already exists!')
    print('Extracted', songDataExtractedCount, 'songs metadata from', playlistName,'.')

def infLoopCheck(killCount,moreSongsToLoad):
    if moreSongsToLoad:
        killCount +=1
        if killCount>10:
            print("Looped 10 times and could not finish looking at playlist data.  Either playlist over 1k songs OR ERROR!")
            return False, killCount
        return True, killCount
    return moreSongsToLoad, killCount

def makeAudioOrFeatureDF(usingAudioAnalysis):
    if usingAudioAnalysis:
        return pd.DataFrame(columns=['song', 'affect', 'playlist'])
    return pd.DataFrame( columns=['song', 'affect', 'playlist', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature' ])



############################################      "MAIN"      #######################################################


playlistID = 'https://open.spotify.com/playlist/3jxJ5VOEudOS5iWfq27deI?si=5a9884f2774348c9'
CSVFile = 'Rock Testing Concat.csv'

affect = ['sad','happy']
#getSongMetadataFromPlaylist(playlistID, affect[1], CSVFile, usingAudioAnalysis=True)
#dont have to use custom csv filename, as shown below!

#getSongMetadataFromPlaylist(playlistID, affect[0], dataSubFolderLocation='Rock')


