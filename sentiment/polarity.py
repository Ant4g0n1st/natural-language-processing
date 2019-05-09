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

def LoadPolarity():
    pol = dict()
    with open(AsResource("polarity.pkl"), "rb") as f:
        pol = pickle.load(f)
    return pol

def Normalize(text):
    global sent_tokenizer
    global d_regex
    global stemMap
    global w_regex
    global tagger 
    global stop
    global pol 

    normalized = []
    p = 0

    for sentence in sent_tokenizer.tokenize(text):

        tokens = [ w.lower() for w in nltk.word_tokenize(sentence) ]

        tokens = list(filter(lambda w : w_regex.fullmatch(w) != None, tokens))
        tokens = list(filter(lambda w : d_regex.search(w) == None, tokens))

        tagged = tagger.tag(tokens)
        tokens = []
    
        for (w, tag) in tagged:
            if tag != None:
                tag = tag.lower().strip()[0]
                w = stemMap.get((w, tag), w)
            if w in stop:
                continue
            if tag != None:
                if (w, tag) in pol:
                    p += pol[(w, tag)] 
            tokens.append(w) 
        
        normalized += tokens

    return normalized, p

# Functions End 

sent_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')

import re

w_regex = re.compile(r'\w+[\n]*')
d_regex = re.compile(r'\d+')

tagger = LoadSpanishTagger()
stop = LoadStopwords()
stemMap = LoadStems()
pol = LoadPolarity()

from bs4 import BeautifulSoup
import glob
import os

corpus_path = os.path.expanduser("~/Desktop/NLP/corpusCine/*.xml")
preprocessed_path = "./preprocessed/"

vocabulary = set()

polarity = []
reviews = []
Y = []

try:
    with open(preprocessed_path + "vocabulary.pkl", "rb") as f:
        vocabulary = pickle.load(f)
    with open(preprocessed_path + "polarity.pkl", "rb") as f:
        polarity = pickle.load(f)
    with open(preprocessed_path + "reviews.pkl", "rb") as f:
        reviews = pickle.load(f)
    with open(preprocessed_path + "labels.pkl", "rb") as f:
        Y = pickle.load(f)
    print("Loaded Reviews/Vocabulary...")
except:
    print("Computing Reviews/Vocabulary...")
    corpusFiles = glob.glob(corpus_path)
    m = len(corpusFiles)
    Y = [ 0 for x in range(0, m) ]
    for k in range(0, m):
        with open(corpusFiles[k], encoding = "latin-1") as f:
            soup = BeautifulSoup(f.read(), "xml")
            
            normal, p = Normalize(soup.get_text())
            
            Y[k] = int(soup.review['rank'])
            reviews.append(normal)
            polarity.append(p)

            vocabulary.update(reviews[-1])
    
    with open(preprocessed_path + "vocabulary.pkl", "wb") as f:
        pickle.dump(vocabulary, f, pickle.HIGHEST_PROTOCOL)
    with open(preprocessed_path + "polarity.pkl", "wb") as f:
        pickle.dump(polarity, f, pickle.HIGHEST_PROTOCOL)
    with open(preprocessed_path + "reviews.pkl", "wb") as f:
        pickle.dump(reviews, f, pickle.HIGHEST_PROTOCOL)
    with open(preprocessed_path + "labels.pkl", "wb") as f:
        pickle.dump(Y, f, pickle.HIGHEST_PROTOCOL)

    print("Done")

vocabulary = sorted(vocabulary)
n = len(vocabulary)
m = len(reviews)
index = dict()

for k in range(0, n):
    index[vocabulary[k]] = k

catPolarity = dict( (k, 0) for k in set(Y) )
catSize = dict( (k, 0) for k in set(Y) )

for k in range(0, m):
    catPolarity[Y[k]] += polarity[k]
    catSize[Y[k]] += 1

for k in list(catSize.keys()):
    catPolarity[k] /= catSize[k]

print("Vocabulary ", n)
print("Reviews ", m)

X = []

try:
    with open(preprocessed_path + "vectors.pkl", "rb") as f:
        X = pickle.load(f)

    print("Loaded Vectors...")
except:
    print("Computing Vectors...")

    X = [ [ 0 for y in range(0, n) ] for x in range(0, m) ] 

    for k in range(0, m):
        for w in reviews[k]:
            X[k][index[w]] += 1

    with open(preprocessed_path + "vectors.pkl", "wb") as f:
        pickle.dump(X, f, pickle.HIGHEST_PROTOCOL)

    print("Done")

from numpy import matrix as NPMatrix
from numpy import array as NPArray

X = NPMatrix(X)
Y = NPArray(Y)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size = 0.2)

model = MultinomialNB() 
model.fit(XTrain, YTrain)
predicted = model.predict(XTest)

#print("Accuracy of prediction is : ", model.score(XTest, YTest) * 100)
#print(classification_report(YTest, predicted))

print(catPolarity)

