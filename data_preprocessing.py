import nltk
nltk.download('punkt')

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
import json
import pickle
import numpy as np

words = []
classes = []
wordTagLlist = []
ignore_words = ['?', '!',',','.', "'s", "'m"]
traindatafile = open('intents.json').read()
intents = json.loads(traindatafile)


def getstemwords(words,ignore_words):
    stemwords = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stemwords.append(w)
    return stemwords

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patternword = nltk.word_tokenize(pattern)
        words.extend(patternword)
        wordTagLlist.append((patternword,intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])
        stemwords = getstemwords(words,ignore_words)

print (stemwords)
print(wordTagLlist[0])
print(classes)

def createbotcorpus(stemwords,classes):
    stemwords = sorted(list(set(stemwords)))
    classes = sorted(list(set(classes)))
    pickle.dump(stemwords,open('words.pkl',"wb"))
    pickle.dump(classes,open("classes.pkl","wb"))
    return(stemwords,classes)

stemwords,classes = createbotcorpus(stemwords,classes)
print(stemwords)
print(classes)
trainingdata = []
numberoftags = len(classes)
labels = [0]*numberoftags
for wordtags in wordTagLlist:
    bagofWords = []
    patternword = wordtags[0]
    for word in patternword:
        index = patternword.index(word)
        word = stemmer.stem(word.lower())
        patternword[index] = word
    for word in stemwords:
        if word in patternword:
            bagofWords.append(1)
        else:
            bagofWords.append(0)
    print(bagofWords)
    
    labelencoding = list(labels)
    tag = wordtags[1]
    tagIndex = classes.index(tag)
    labelencoding[tagIndex] = 1
    trainingdata.append([bagofWords,labelencoding])
print(trainingdata[0])

def preproccestraindata(trainingdata):
    trainingdata = np.array(trainingdata,dtype = object)
    trainx = list(trainingdata[:,0])
    trainy = list(trainingdata[:,1])
    print(trainx[0])
    print(trainx[0])
    return(trainx,trainy)
trainx,trainy = preproccestraindata(trainingdata)
