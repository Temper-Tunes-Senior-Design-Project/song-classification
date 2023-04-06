
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle
#this folder location will be changed to DB location in actual cloud function
from os import chdir
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from bs4 import BeautifulSoup
from bs4 import element
from copy import deepcopy
import re

#________________________________________________________________________________________________________________


#COMBINED MODEL OUTLINE -- 
# inputs:  sp_user( aka the client!!!), UID
#^note: we could pass in the user's last login date, instead of UID if we waned to, 
# as this would provide beter separation of concerns of updating last login date vs the function 
# thats supposed to be getting the song moods

#outputs: verifies success or failure of the function

from datetime import datetime

def get_user_song_moods_advanced(sp_user,UID):

        #0. get the track ids of unlabelled songs from the list of new liked songs from the user
        last_login = DB.get_last_login(UID) #IF NONE, return ''

        #If last login is a string, we need to convert it to a utc datetime object
        #if isinstance(last_login, str):
        #        last_login = datetime.strptime(last_login, '%Y-%m-%d %H:%M:%S.%f')
        
        track_ids = retrieveTrackIds(sp_user,last_login)

        remaining_track_ids = DB.check_if_already_labelled(track_ids)

        if len(remaining_track_ids) == 0:
                DB.update_user_liked_songs(UID,track_ids)
                DB.update_last_login(UID,datetime.utcnow())
                return "success"

        #1. get the song features of unlabelled songs

        featuresDF = retrieveTrackFeatures(remaining_track_ids) #this should drop the ids of songs that dont have features
        featuresDict = clipAndNormalize(featuresDF)#.set_index('id').T.to_dict('list')

        #2. if possible, get the lyrics of the songs

        #make a dictionary of song titles and artist names
        scraperInputs = getTitlesAndArtists(sp_user, remaining_track_ids)


        #-------------------------------------------------
        #compute gets expensive here, so just do a for loop on remaining_track_ids
        #where u 
        # 1. get lyrics, 
        # 2. check if features are available as well 
        # 3. make a prediction, based on 1 and 2 
        # 4. immediately update the database and last added_at date, in case of a time out
        #-------------------------------------------------
        all_lyrics_dict = getScrapedLyrics(scraperInputs)

        #MAYBE Tokenize THE LYRICS HERE (use batch processing?!) 
        #would need to verify compute restrictions of cloud function first!

        overlap_keys = [key for key in featuresDict.keys() if key in all_lyrics_dict.keys()]
        only_features = [key for key in featuresDict.keys() if key not in overlap_keys]
        only_lyrics = [key for key in all_lyrics_dict.keys() if key not in overlap_keys]


        #3 get predictions and update database

        #3a. if there are no features or lyrics, return an error
        if len(featuresDict.keys()) == 0 and len(all_lyrics_dict.keys()) == 0:
                #probably just return a flag that says there are no features or lyrics not the response here
                return "songs found but no data available for predictions"
        
        else:
                predictions = {}
                #chdir('C:/Users/mlar5/OneDrive/Desktop/Code Folder/198 Senior Design/Models/Spotipy')


                #for first version, tokenize the lyrics and then pass then to the model inside the for loop

                for key in overlap_keys: #- could probably be done in batches regardless
                        MLP_pred, MLP_pred_probability = getMoodLabelMLP(featuresDict[key])

                        #BERT_pred, BERT_pred_probability = getMoodLabelBERT(BERT_model,all_lyrics_dict[key])
                        BERT_pred, MLP_flag = getOnlyMoodLabelFromLyrics(all_lyrics_dict[key])
                        if MLP_pred == BERT_pred:
                                prediction = MLP_pred
                        else:
                                #add probabilities and choose the label with the highest probability
                                sum_probabilities = MLP_pred_probability + BERT_pred_probability
                                prediction = np.argmax(sum_probabilities)
                        predictions[key]=prediction

                for key in only_features:
                        MLP_pred, MLP_pred_probability = getMoodLabelMLP(featuresDict[key])
                        predictions[key]=MLP_pred

                for key in only_lyrics:
                        #BERT_pred, BERT_pred_probability = getMoodLabelBERT(BERT_model,all_lyrics_dict[key])
                        BERT_pred, MLP_flag = getOnlyMoodLabelFromLyrics(all_lyrics_dict[key])
                        predictions[key]=BERT_pred

                DB.add_song_moods(predictions)
                
                DB.update_user_liked_songs(UID,predictions.keys()) 
                # ^^Need to add a check to discard songs no longer on spotify, 
                # otherwise we might recommend songs that are no longer on spotify

                DB.update_last_login(UID,datetime.utcnow())

        return "success"

#________________________________________________________________________________________________________________
### Helper Functions of get_User_Song_Moods_2_models
#________________________________________________________________________________________________________________

def retrieveTrackIds(sp, user_prior_login_date):
    track_ids = []
    offset = 0
    limit = 50
    liked_tracks = sp.current_user_saved_tracks(limit=limit, offset=offset)
    endLoopEarly = False
    while True:
        for item in liked_tracks['items']:
            if user_prior_login_date != '':
                if datetime.strptime(item['added_at'],'%Y-%m-%dT%H:%M:%SZ')> user_prior_login_date:
                    track_ids.append(item['track']['id'])
                else:
                    endLoopEarly = True
                    break
            else:
                track_ids.append(item['track']['id'])
        offset += limit
        
        if len(liked_tracks['items']) < limit or endLoopEarly:
            # All tracks have been retrieved
            break
        
        liked_tracks = sp.current_user_saved_tracks(limit=limit, offset=offset)
    
    return track_ids


'''def getSongFeaturesToDict(track_ids):
        featuresDict = {}
        for i in range(0,len(track_ids),100):
                features = sp.audio_features(track_ids[i:i+100])
                for feature in features:
                        if feature != None:
                                featuresDict[feature['id']]=feature
        return featuresDict'''

def retrieveTrackFeatures(sp, track_ids):
    dfs = []
    for i in range(0, len(track_ids), 50):
        # Retrieve track features with current offset
        current_features = sp.audio_features(track_ids[i:i+50])
        
        # Convert to DataFrame
        df = pd.DataFrame(current_features)
        
        # Remove columns that we don't need
        df = df.drop(['type', 'uri', 'analysis_url', 'track_href'], axis=1)
        
        
        # Append to list of dataframes
        dfs.append(df)
    
    # Concatenate all dataframes into a single one
    features_df = pd.concat(dfs, ignore_index=True)
    
    
    #convert to dictionary, with track id as key
    #featuresDict = features_df.set_index('id').T.to_dict('list')
    
    return features_df

def clipAndNormalize(features):
    #clip the features to the range of the training data
    features['danceability'] = features['danceability'].clip(lower=0.22718080000000002, upper=0.906)
    features['energy'] = features['energy'].clip(lower=0.03545904, upper=0.978)
    features['loudness'] = features['loudness'].clip(lower=-26.4981552, upper=-1.6015904000000007)
    features['speechiness'] = features['speechiness'].clip(lower=0.0257, upper=0.46640959999999926)
    features['acousticness'] = features['acousticness'].clip(lower=8.353136000000001e-05, upper=0.9884095999999992)
    features['instrumentalness'] = features['instrumentalness'].clip(lower=0.0, upper=0.956)
    features['liveness'] = features['liveness'].clip(lower=0.0494, upper=0.697)
    features['valence'] = features['valence'].clip(lower=0.0382, upper=0.923)
    features['tempo'] = features['tempo'].clip(lower=63.7631808, upper=188.00344319999996)
    features['duration_ms'] = features['duration_ms'].clip(lower=88264.8768, upper=372339.1727999991)
    features['time_signature'] = features['time_signature'].clip(lower=3.0, upper=5.0)
    
    #normalize the data
    scaler = pickle.load(open('scaler1.pkl', 'rb'))
    #fit on all columns except the track id
    rawfeatures = features.drop(['id'], axis=1)
    preprocessedFeatures = scaler.transform(rawfeatures)

    #convert to dictionary, with track id as key
    preprocessedFeatures = pd.DataFrame(preprocessedFeatures, columns=rawfeatures.columns)
    preprocessedFeatures['id']= features['id']
    preprocessedFeatures = preprocessedFeatures.set_index('id').T.to_dict('list')
    return preprocessedFeatures


def getTitlesAndArtists(sp, track_ids):
    titleArtistPairs = {}
    for i in range(0,len(track_ids),50):
        tracks = sp.tracks(track_ids[i:i+50])
        for track in tracks['tracks']:
            title=track['name']
            #check if the track ends with (feat. artist) using a regex
            if re.search(r' \(feat. .*\)$', title):
                #remove the (feat. artist) from the title
                title = re.sub(r' \(feat. .*\)$', '', title)

            artists=[]
            for artist in track['artists']:
                artists.append(artist['name'])
            titleArtistPairs[track['id']] = {'title':title,'artist(s)':artists}

    return titleArtistPairs

def getScrapedLyrics(scraperInputs):
        all_lyrics_dict = {}
        for id, songInfo in scraperInputs.items():
                #maybe add a sleep or something to prevent getting blocked
                lyrics = scrapeLyrics(songInfo['artist(s)'],songInfo['title'])
                if len(lyrics) > 0:
                        all_lyrics_dict[id]=lyrics
        return all_lyrics_dict


def getMoodLabelMLP(songFeautures):
        with open('MLP1.pkl','rb') as f:
            model = pickle.load(f)
        prediction = model.predict(songFeautures)
        pred_probability=model.predict_proba(songFeautures)
        return prediction, pred_probability
#________________________________________________________________________________________________________________
#                 BERT Sentiment Analysis Hard-coded info that needs to be used somewhere in the cloud
#                (or just saved to a file and loaded in the cloud function)
#________________________________________________________________________________________________________________

model = AutoModelForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")
tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
#^^these can be pickled and loaded in the future for the cloud OR called to from an endpoint

emotionsAsValenceArousal= { 'admiration':(.6,.4),'amusement':(.6,.2),'anger':(-.8,.6),'annoyance':(-.6,.6),'approval':(.8,.6),'caring':(.6,-.2),'confusion':(-.2,.2),'curiosity':(0,.4),'desire':(.6,.6),'despair':(-.8,-.6),'disappointment':(-.6,-.6),'disapproval':(-.8,.65),'disgust':(-.8,.2),'embarrassment':(-.6,.4),'envy':(-.6,.4),'excitement':(.6,.8),'fear':(-.6,.8),'gratitude':(.6,-.6),'grief':(-.6,-.8),'gratitude':(.6,-.6),'grief':(-.6,-.8),'joy':(.8,.2),'love':(.8,.4),'nervousness':(-.4,.6),'optimism':(.6,.2),'pride':(.6,.1),'realization':(.2,.2),'relief':(.4,-.4),'remorse':(-.6,-.4),'sadness':(-.8,-.2),'surprise':(.2,.6),'neutral':(0,0)}

emotion_dict = model.config.id2label

#________________________________________________________________________________________________________________
#                 BERT Sentiment Analysis Functions
#________________________________________________________________________________________________________________



def getOnlyMoodLabelFromLyrics(lyrics, emotion_dict=emotion_dict, emotionsAsValenceArousal=emotionsAsValenceArousal,printValenceArousal=False): #model=model, tokenizer=tokenizer,
    #device = 'cuda' if cuda.is_available() else 'cpu'
    
    #load in local copy of BERT model with its local path INSTEAD OF THIS url (which is slow)
    BERT_model = AutoModelForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")
    BERT_Tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
    mood,relyOnLinearModel = getMoodLabelFromLyrics(lyrics,BERT_model, BERT_Tokenizer, emotion_dict, emotionsAsValenceArousal, device='cpu',printValenceArousal=printValenceArousal)
    return mood,relyOnLinearModel


def getMoodLabelFromLyrics(lyrics,model, tokenizer, emotion_dict, emotionsAsValenceArousal,printValenceArousal = False,disregardNeutral=True, printRawScores=False, printTopN=False,topScoresToPrint=3,max_length=512, device="cuda",  returnSongSegments=False):
    relyOnLinearResults = False
    softmaxScoresPerHeader = {}
    model.to(device)
    
    #part 1 - break up the lyrics into chunks and get the tokens
    if returnSongSegments:
        songTokenChunks,freqs,songSegs =breakUpSongByHeaders(lyrics,tokenizer,returnSongSegments=returnSongSegments,max_length=max_length, device=device)
    else:
        songTokenChunks,freqs =breakUpSongByHeaders(lyrics,tokenizer,returnSongSegments=returnSongSegments,max_length=max_length, device=device)

    #part 2 - get the softmax score for each chunk

    if len(songTokenChunks) == 1:
        disregardNeutral=False

    #softmax scores returns COMBINED SINGLE LABEL -- MAYBE TRY MULTIPLE LABELS AND TAKE THE MOST COMMON
    for header,tokenChunksPerHeaders in songTokenChunks.items():
        for tokenChunk in tokenChunksPerHeaders:
            ## ^^ If I encode multiple songs in batches, then I would make another for loop here and not just use tokenChunk[0]
            ## but it might be too complicated to do that this way.  
            # I'd have to make a function that breaks up the lyrics into chunks, 
            # and then return the chunks in a way that we still know which chunk belongs to which song and header
            if header not in softmaxScoresPerHeader:
                softmaxScoresPerHeader[header] = getSoftmax(model,tokenizer,tokens=tokenChunk[0],n=topScoresToPrint, printTopN=printTopN, printRawScores=printRawScores,device=device)
            else:
                softmaxScoresPerHeader[header] += getSoftmax(model,tokenizer,tokens=tokenChunk[0],n=topScoresToPrint, printTopN=printTopN, printRawScores=printRawScores,device=device)
            
            
    #Part 3 determine what to do with the neutral labels
    moodLabel, valence, arousal = convertScoresToLabels(softmaxScoresPerHeader,freqs, emotionsAsValenceArousal,emotion_dict,disregardNeutral=disregardNeutral,printValenceArousal=printValenceArousal)

    if moodLabel=='top ratings all neutral':
        disregardNeutral=False
        moodLabel, valence, arousal = convertScoresToLabels(softmaxScoresPerHeader,freqs, emotionsAsValenceArousal,emotion_dict,disregardNeutral=disregardNeutral,printValenceArousal=printValenceArousal)
        relyOnLinearResults = True
    if moodLabel=='neutral' or (-0.1<valence<0.1 and -0.1<arousal<0.1):
        relyOnLinearResults = True
    #part 4 - return the most common label
    return moodLabel, relyOnLinearResults


# input: a string of whole song
# output: a dictionary of with header values and a list of tensors (sometmes more than 1 item) for each header chunk
def breakUpSongByHeaders(songLines, tokenizer, max_length=512, device="cuda",  returnSongSegments=False):
    songSegmentsDict = {}
    tokenSegmentsDict = {}
    headerFreqsDict = {}

    #strip the trailing whitespace
    lines = [line.strip() for line in songLines]

    #find the lines that start with [ and end with ]
    headerLinesIndex = [i for i, line in enumerate(lines) if line.startswith('[') and line.endswith(']')]
    #check for consecutive headers indexes and remove the first one
    for i in range(len(headerLinesIndex)-1):
        if headerLinesIndex[i+1] - headerLinesIndex[i] == 1:
            headerLinesIndex[i] = -1
    headerLinesIndex = [i for i in headerLinesIndex if i != -1]

    for i in range(len(headerLinesIndex)):
        header_line = lines[headerLinesIndex[i]][1:-1]  # remove square brackets
        if header_line in songSegmentsDict:
            songSegmentsDict[header_line][0] += 1
        elif i == len(headerLinesIndex)-1:
            songSegmentsDict[header_line] = [1, " ".join(lines[headerLinesIndex[i]+1:]), lines[headerLinesIndex[i]+1:]]
        else:
            songSegmentsDict[header_line] = [1, " ".join(lines[headerLinesIndex[i]+1:headerLinesIndex[i+1]]), lines[headerLinesIndex[i]+1:headerLinesIndex[i+1]]]

    for header, lyrics in songSegmentsDict.items():
        if returnSongSegments:
            tokenSegmentsDict[header],subLyrics = breakUpLargeLyricChunks(lyrics[1],lyrics[2],tokenizer,returnLyricsSegments=returnSongSegments,max_length=max_length, device=device)
            songSegmentsDict[header]=subLyrics
        else:
            tokenSegmentsDict[header] = breakUpLargeLyricChunks(lyrics[1],lyrics[2],tokenizer,returnLyricsSegments=returnSongSegments,max_length=max_length, device=device)
        headerFreqsDict[header] = lyrics[0]

    if returnSongSegments:
        return tokenSegmentsDict,headerFreqsDict,songSegmentsDict
    else:
        return tokenSegmentsDict,headerFreqsDict




def breakUpLargeLyricChunks(lyricsChunkString, lines,tokenizer, max_length=512, device="cuda", returnLyricsSegments=False):
    #lines = lyricsChunkString.splitlines()  # split the lyrics into lines
    segments = []  # store the lyrics segments
    token_segments = []  # store the tokenized segments as tensors

    token_segment = tokenizer.encode(lyricsChunkString, return_tensors="pt")#.to(device)

    if len(token_segment[0]) <= max_length:
        token_segment = token_segment.unsqueeze(0)
        token_segments.append(token_segment)
        segments.append(lyricsChunkString)
    else:
        # calculate the average number of lines per segment. Add +2 to ensure segments are not still too long
        avg_lines_per_segment = len(lines) // ((len(token_segment[0]) // max_length) + 2)

        # loop through the lines and group them into segments of roughly the same length
        for start_idx in range(0, len(lines), avg_lines_per_segment):
            end_idx = start_idx + avg_lines_per_segment

            smallLastChunk = end_idx >= len(lines)-2
            
            if smallLastChunk:
                segment = " ".join(lines[start_idx:])
            else:
                segment = " ".join(lines[start_idx:end_idx])
            segments.append(segment)

            # tokenize the segment and convert to tensor
            token_segment = tokenizer.encode(segment, return_tensors="pt")#.to(device)
            token_segment = token_segment.unsqueeze(0)
            token_segments.append(token_segment)
            #NOTE: ^^ If I use batch_encode_plus, I can get the tokenized segments as a list of tensors in one step
            #I would just have to do it after the loop. 
            #Since it is a small list though, I don't think it will make a difference in this case

            if smallLastChunk:
                #this is the last segment early, so break out of the loop
                break

    if returnLyricsSegments:  
        return token_segments, segments
    else:
        return token_segments


def getSoftmax(model,tokenizer, tokens = None, sentence=None, n=3,printRawScores=False, printTopN=False,device='cuda'):
    if tokens is None:
        tokens = tokenizer.encode(sentence, return_tensors="pt")
    if device=='cuda':
        tokens = tokens.cuda()
    result = model(tokens)
    emotion = result.logits
    emotion = emotion.cpu().detach().numpy()
    emotion = emotion[0]
    softmax = tf.nn.softmax(emotion)
    #convert to numpy array
    softmax = softmax.numpy()
    if printRawScores:
        print(softmax)
    
    if printTopN:
        emotion = emotion.argsort()[-n:][::-1]
        emotion = emotion.tolist()
        printTopEmotions(emotion,model, softmax)
    return softmax


def printTopEmotions(emotion, model, softmax):
    
    #identify the label of top n emotions from emotion list
    #softmax is in the order of the values in emotion_dict so we can use emotion[id] to get the softmax value
    id=0
    emotion_dict = model.config.id2label
    for i in emotion:
        print(emotion_dict[i])
        print(softmax[emotion[id]]*100,"%")
        id+=1
    return


def convertScoresToLabels(softmaxScoresPerHeader,headerFreqs, emotionsAsValenceArousal,emotion_dict,disregardNeutral = True, printValenceArousal=False,printTopChunkEmotions=False):
    #convert the softmax scores to a valence and arousal score
    #softmax scores are in the order of the values in emotion_dict so we can use emotion[id] to get the softmax value
    valence=0
    arousal=0
    softmaxScoresApplied=0
    #find the key in emotion_dict that corresponds to neutral
    neuturalKey = [key for key, value in emotion_dict.items() if value == 'neutral'][0]
    for key, softmaxScores in softmaxScoresPerHeader.items():
        #check if neutral is the highest softmax score
        if disregardNeutral and neuturalKey==softmaxScores.argmax():
            continue
        else:
            #multiply the softmax score by the valence and arousal values and add to the total valence and arousal
            #do this for the number in the headerFreqs dictionary
            for i in range(headerFreqs[key]):
                id=0
                softmaxScoresApplied+=1
                for i in softmaxScores:
                    valence+=i*emotionsAsValenceArousal[emotion_dict[id]][0]
                    arousal+=i*emotionsAsValenceArousal[emotion_dict[id]][1]
                    id+=1
    #divide the total valence and arousal by the number of softmax scores applied
    if softmaxScoresApplied!=0:
        valence=valence/softmaxScoresApplied
        arousal=arousal/softmaxScoresApplied
        mood =determineMoodLabel(valence,arousal,printValenceArousal=printValenceArousal)
        return mood, valence, arousal
    else:
        return 'top ratings all neutral', valence, arousal
    #note this means all top chunk emotions were neutral as opposed to true neutral where all emotions balance out to neutral

def determineMoodLabel(valence,arousal,printValenceArousal=False):
    #determine the diagonal of the circumplex model that the valence and arousal scores fall on
    #MAKE 2 BOXES OF THE CIRCUMPLEX MODEL A MOOD 

    energetic =   -0.5<valence<0.5 and arousal>0.5
    happy =       valence>0.5 and -.5<arousal<0.5
    calm =       -0.5<valence<0.5 and arousal<-0.5
    sad =         valence<-0.5 and -.5<arousal<0.5

    excited =   not (happy or energetic) and valence>0 and arousal>0
    content =   not (calm or happy) and valence>0 and arousal<0
    depressed = not (calm or sad) and valence<0 and arousal<0
    angry =   not (energetic or sad) and valence<0 and arousal>0


    if energetic:
        mood='energetic'
    elif happy:
        mood='happy'
    elif calm:
        mood='calm'
    elif sad:
        mood='sad'
    elif excited:
        mood='excited'
    elif content:
        mood='content'
    elif depressed:
        mood='depressed'
    elif angry:
        mood='angry'
    else:
        mood='neutral'
    
    if printValenceArousal:
        print("Valence: ",valence)
        print("Arousal: ",arousal)
    return mood     



#________________________________________________________________________________________________________________
#                  SCRPAER CODE
#________________________________________________________________________________________________________________

#Helps parse miscellaneous tags <i>, </br>, etc,.
def _lyricsHelper(html, lyrics_list):
    for tag in html.childGenerator():
        if type(tag) == element.NavigableString:
            _handleLyricAppend(lyrics_list, tag.text.strip())
        elif tag.name == 'br' and lyrics_list[len(lyrics_list) - 1] != '':
            lyrics_list.append('')
        elif html.name == 'a':
            _lyricsHelper(tag, lyrics_list)

#Reads the HTML for lyrics dividers (if they exist) and appends the lyrics line by line to a list
def _getLyricsFromHTML(html):
    lyrics = html.findAll("div", {"data-lyrics-container" : "true"})
    lyrics_list = ['']
    for segment in lyrics:
        for a in segment.childGenerator():
            lyric = None
            if type(a) == element.NavigableString:
                lyric = a.strip()
                _handleLyricAppend(lyrics_list, lyric)
            else:
                _lyricsHelper(a, lyrics_list)
            if a.name == 'br' and lyrics_list[len(lyrics_list) - 1] != '':
                lyrics_list.append('')
    return lyrics_list

#Helper function to handle appending and manipulating lyrics_list. A new line is generated only for </br> tags
def _handleLyricAppend(lyrics_list, lyric):
    if lyric is not None:
        last_index = len(lyrics_list) - 1
        #Handle special cases (parenthesis and symbols stick with words for instance)
        if lyrics_list[last_index] != '' and (lyrics_list[last_index][-1] in ['(','[','{','<'] or lyric in [')',']','}','>','!','?',',','.']):
            lyrics_list[last_index] += lyric
        else:
            lyrics_list[last_index] += " " + lyric if lyrics_list[last_index] != '' else lyric

#Determines whether a song will need to be translated (returns the link if it does, otherwise returns None)
def _getSongTranslationLink(html):
    translation_tags = html.findAll('a', class_='TextButton-sc-192nsqv-0 hVAZmF LyricsControls__TextButton-sghmdv-6 hXTHjZ')
    for tag in translation_tags:
        if "english-translations" in tag['href']:
            return tag['href']
    return None

#Determines whether a page exists
def _pageExists(html):
    return html.find('div', class_='render_404') == None
        
#function to scrape lyrics from genius, takes an array of artists, and songname
def scrapeLyrics(artistnames, songname):
    lyrics_list = []
    found = False
    i = 0
    html = None
    while i < len(artistnames) and not(found):
        artistname = artistnames[i]
        artistname2 = str(artistname.replace(' ','-')) if ' ' in artistname else str(artistname)
        songname2 = str(songname.replace(' ','-')) if ' ' in songname else str(songname)
        page_url = 'https://genius.com/'+ artistname2 + '-' + songname2 + '-' + 'lyrics'
        page = requests.get(page_url)
        html = BeautifulSoup(page.text, 'html.parser') 
        found = _pageExists(html)
        i += 1
    if found:
        #check if there is an english translation
        translation_url = _getSongTranslationLink(html)
        if translation_url is not None:
            page = requests.get(translation_url)
            html = BeautifulSoup(page.text, 'html.parser') 
            lyrics_list = _getLyricsFromHTML(html)
        else:
            #If there isn't a translation, make sure it's in english in the first place
            english = False
            for script in html.findAll('script'):
                if "language\\\":\\\"en" in str(script):
                    english = True
            if english: lyrics_list = _getLyricsFromHTML(html)
    return lyrics_list
    


#________________________________________________________________________________________________________________
#                  MAIN
#________________________________________________________________________________________________________________

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
