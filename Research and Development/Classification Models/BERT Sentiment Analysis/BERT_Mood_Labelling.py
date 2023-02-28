import tensorflow as tf
from torch import cuda

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")
tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
#^^these can be pickled and loaded in the future for the cloud OR called to from an endpoint

emotionsAsValenceArousal= { 'admiration':(.6,.4),'amusement':(.6,.2),'anger':(-.8,.6),'annoyance':(-.6,.6),'approval':(.8,.6),'caring':(.6,-.2),'confusion':(-.2,.2),'curiosity':(0,.4),'desire':(.6,.6),'despair':(-.8,-.6),'disappointment':(-.6,-.6),'disapproval':(-.8,.65),'disgust':(-.8,.2),'embarrassment':(-.6,.4),'envy':(-.6,.4),'excitement':(.6,.8),'fear':(-.6,.8),'gratitude':(.6,-.6),'grief':(-.6,-.8),'gratitude':(.6,-.6),'grief':(-.6,-.8),'joy':(.8,.2),'love':(.8,.4),'nervousness':(-.4,.6),'optimism':(.6,.2),'pride':(.6,.1),'realization':(.2,.2),'relief':(.4,-.4),'remorse':(-.6,-.4),'sadness':(-.8,-.2),'surprise':(.2,.6),'neutral':(0,0)}

emotion_dict = model.config.id2label

def getOnlyMoodLabelFromLyrics(lyrics,model=model, tokenizer=tokenizer, emotion_dict=emotion_dict, emotionsAsValenceArousal=emotionsAsValenceArousal,printValenceArousal=False):
    device = 'cuda' if cuda.is_available() else 'cpu'
    mood,relyOnLinearModel = getMoodLabelFromLyrics(lyrics,model, tokenizer, emotion_dict, emotionsAsValenceArousal, device=device,printValenceArousal=printValenceArousal)
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
def breakUpSongByHeaders(fullSongLyricsString, tokenizer, max_length=512, device="cuda",  returnSongSegments=False):
    songSegmentsDict = {}
    tokenSegmentsDict = {}
    headerFreqsDict = {}

    #split the song into a list of lines
    lines = fullSongLyricsString.splitlines()
    #strip the trailing whitespace
    lines = [line.strip() for line in lines]

    #find the lines that start with [ and end with ]
    headerLinesIndex = [i for i, line in enumerate(lines) if line.startswith('[') and line.endswith(']')]

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
    #print(type(lyricsChunkString))
    token_segment = tokenizer.encode(lyricsChunkString, return_tensors="pt").to(device)

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
            token_segment = tokenizer.encode(segment, return_tensors="pt").to(device)
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
    relaxed =   not (calm or happy) and valence>0 and arousal<0
    depressed = not (calm or sad) and valence<0 and arousal<0
    anxious =   not (energetic or sad) and valence<0 and arousal>0


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
    elif relaxed:
        mood='relaxed'
    elif depressed:
        mood='depressed'
    elif anxious:
        mood='anxious'
    else:
        mood='neutral'
    
    if printValenceArousal:
        print("Valence: ",valence)
        print("Arousal: ",arousal)
    return mood     


fullLyricsGoEasyOnMe = """
[Verse 1]
There ain't no gold in this river
That I've been washin' my hands in forever
I know there is hope in these waters
But I can't bring myself to swim
When I am drowning in this silence
Baby, let me in

[Chorus]
Go easy on me, baby
I was still a child
Didn't get the chance to
Feel the world around me
I had no time to choose what I chose to do
So go easy on me
[Verse 2]
There ain't no room for things to change
When we are both so deeply stuck in our ways
You can't deny how hard I've tried
I changed who I was to put you both first
But now I give up

[Chorus]
Go easy on mе, baby
I was still a child
Didn't get the chance to
Feel thе world around me
Had no time to choose what I chose to do
So go easy on me

[Bridge]
I had good intentions
And the highest hopes
But I know right now
It probably doesn't even show

[Chorus]
Go easy on me, baby
I was still a child
I didn't get the chance to
Feel the world around me
I had no time to choose what I chose to do
So go easy on me
"""



print(getOnlyMoodLabelFromLyrics(fullLyricsGoEasyOnMe,printValenceArousal=True))
