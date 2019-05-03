import pickle
import nltk

# Functions Begin

resources_path = "../resources/"

def AsResource(fileName):
    return resources_path + fileName

# Load the spanish POS tagger from
# the file 'spanish-tagger.pkl' 
def LoadSpanishTagger():
    f = open(AsResource("spanish-tagger.pkl"), "rb")
    tagger = pickle.load(f)
    f.close()
    
    return tagger

# Load the stem map as a dictionary
# from the file 'stems.txt'
def LoadStems():
    f = open(AsResource("lemmas.pkl"), "rb")
    stemMap = pickle.load(f)
    f.close()

    return stemMap

# Load the stopwords as a set from
# the file 'stop.txt'
def LoadStopwords():
    f = open(AsResource("stop.txt"))
    stop = set(f.read().split())
    f.close()

    return stop

def Normalize(text):
    global sent_tokenizer
    global d_regex
    global stemMap
    global w_regex
    global tagger 
    global stop

    normalized = []

    for sentence in sent_tokenizer.tokenize(text):

        tokens = [ w.lower() for w in nltk.word_tokenize(sentence) ]

        tokens = list(filter(lambda w : w_regex.fullmatch(w) != None, tokens))
        tokens = list(filter(lambda w : d_regex.search(w) == None, tokens))

        tagged = tagger.tag(tokens)
        tokens = []
    
        for (w, tag) in tagged:
            if tag != None:
                tag = tag.lower()[0]
                w = stemMap.get((w, tag), w)
            if w in stop:
                continue
            tokens.append(w) 
        
        normalized += tokens

    return normalized 

# Functions End 

sent_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')

import re

w_regex = re.compile(r'\w+[\n]*')
d_regex = re.compile(r'\d+')

tagger = LoadSpanishTagger()
stop = LoadStopwords()
stemMap = LoadStems()

from bs4 import BeautifulSoup
import glob
import os

corpus_path = os.path.expanduser("~/Desktop/NLP/corpusCine/*.xml")

vocabulary = set()

corpusFiles = glob.glob(corpus_path)
m = len(corpusFiles)

Y = [ 0 for x in range(0, m) ]
reviews = []

print("COMPUTING VOC")

for k in range(0, m):
    with open(corpusFiles[k], encoding = "latin-1") as f:
        soup = BeautifulSoup(f.read(), "xml")

        reviews.append(Normalize(soup.get_text()))
        Y[k] = int(soup.review['rank'])

        vocabulary.update(reviews[-1])

print("DONE")

vocabulary = sorted(vocabulary)
n = len(vocabulary)
index = dict()

for k in range(0, n):
    index[vocabulary[k]] = k

print("ART ", m)
print("VOC", n)

print("COMPUTING VECTORS")

X = [ [ 0 for y in range(0, n) ] for x in range(0, m) ] 

for k in range(0, m):
    for w in reviews[k]:
        X[k][index[w]] += 1

print("DONE")

print(vocabulary[:10])

import numpy as np

X = np.matrix(X)
Y = np.array(Y)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 

from sklearn.naive_bayes import MultinomialNB

XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size = 0.2)

model = MultinomialNB() 
model.fit(XTrain, YTrain)
predicted = model.predict(XTest)

print("Accuracy of prediction is : ", model.score(XTest, YTest) * 100)
print("Confussion Matrix:\n", confusion_matrix(YTest, predicted))
