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
preprocessed_path = "./preprocessed/"

vocabulary = set()

reviews = []
Y = []

try:
    with open(preprocessed_path + "vocabulary.pkl", "rb") as f:
        vocabulary = pickle.load(f)
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

            reviews.append(Normalize(soup.get_text()))
            Y[k] = int(soup.review['rank'])

            vocabulary.update(reviews[-1])
    
    with open(preprocessed_path + "vocabulary.pkl", "wb") as f:
        pickle.dump(vocabulary, f, pickle.HIGHEST_PROTOCOL)
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

from sklearn.metrics import classification_report
from mord import LogisticIT 

XTest = None
model = None
YTest = None

try:
    with open(preprocessed_path + "xtest.pkl", "rb") as f:
        XTest = pickle.load(f)
    with open(preprocessed_path + "ytest.pkl", "rb") as f:
        YTest = pickle.load(f)
    with open(preprocessed_path + "model.pkl", "rb") as f:
        model = pickle.load(f)

    print("Loaded Model...")
except:
    print("Training Model...")

    from sklearn.model_selection import train_test_split

    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size = 0.2)
    model = LogisticIT(max_iter = 100) 
    model.fit(XTrain, YTrain)

    with open(preprocessed_path + "xtest.pkl", "wb") as f:
        pickle.dump(XTest, f, pickle.HIGHEST_PROTOCOL)
    with open(preprocessed_path + "ytest.pkl", "wb") as f:
        pickle.dump(YTest, f, pickle.HIGHEST_PROTOCOL)
    with open(preprocessed_path + "model.pkl", "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    print("Done")

predicted = model.predict(XTest)

print("Accuracy of prediction is : ", model.score(XTest, YTest) * 100)
print(classification_report(YTest, predicted))
