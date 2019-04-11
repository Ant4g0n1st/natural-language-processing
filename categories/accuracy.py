from bs4 import BeautifulSoup
import pickle
import numpy
import math
import nltk
import re

# Functions Begin

# Read only the text from an HTML file.
def ReadTextFromHTML(fileName):
    f = open(fileName)
    soup = BeautifulSoup(f.read(), 'html.parser')
    f.close()

    return soup.get_text()

# Load the stopwords as a set from
# the file 'stop.txt'
def LoadStopwords():
    f = open('stop.txt')
    stop = set(f.read().split())
    f.close()

    return stop

# Load the stem map as a dictionary
# from the file 'stems.txt'
def LoadStems():
    f = open("lemmas.en.pickle", "rb")
    stemMap = pickle.load(f)
    f.close()

    return stemMap

# Load the spanish POS tagger from
# the file 'spanish-tagger.pkl' 
def LoadSpanishTagger():
    f = open('spanish-tagger.pkl', 'rb')
    tagger = pickle.load(f)
    f.close()
    
    return tagger

# Functions End 

stemMap = LoadStems()

examples = []
label = []

f = open("spam.txt", "r", encoding = "latin-1")
for line in f:
    tokens = nltk.word_tokenize(line.strip())
    if len(tokens) == 0:
        continue
    examples.append(tokens[ : -2])
    if tokens[-1] == "spam":
        label.append(0) 
    else: 
        label.append(1)
f.close()

m = len(examples)
tokens = []

for k in range(0, m):
    z = len(examples[k])
    for l in range(0, z):
        w = examples[k][l].strip()
        isUp = w.isupper()
        w = w.lower()
        if w in stemMap:
            w = stemMap[w]
        if isUp:
            w = w.upper()
        examples[k][l] = w
    tokens = tokens + examples[k] 

vocabulary = sorted(set(tokens))
n = len(vocabulary)

index = dict()

for k in range(0, n):
    index[vocabulary[k]] = k

freq = [ [ 0 for x in range(0, n) ] for y in range(0, m) ] 
docFreq = [ 0 for x in range(0, n) ]

for k in range(0, m):
    z = len(examples[k])
    for l in range(0, z):
        docFreq[index[examples[k][l]]] += 1
        freq[k][index[examples[k][l]]] += 1

idfFreq = [ [ 0 for x in range(0, n) ] for y in range(0, m) ] 
idf = [ math.log((m + 1) / x) for x in docFreq ]

for k in range(0, m):
    z = len(examples[k])
    for l in range(0, z):
        idfFreq[k][l] = math.log1p(freq[k][l]) * idf[l]

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 

from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt

nSpam = label.count(0)

idfScores = []
scores = []
ticks = []

for k in range(2 * nSpam, len(label) + 1, 50):

    X = numpy.matrix(freq[0 : k])
    y = numpy.array(label[0 : k])

    ticks.append(k - 2 * nSpam)

    XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size = 0.2)
    model = MultinomialNB()
    model.fit(XTrain, YTrain)
    scores.append(model.score(XTest, YTest) * 100)
    #model.fit(X, y)
    #scores.append(model.score(X, y) * 100)

"""
    X = numpy.array(idfFreq[0 : k])

    XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size = 0.2)
    model = MultinomialNB()
    model.fit(XTrain, YTrain)
    idfScores.append(model.score(XTest, YTest) * 100)
"""

#plt.plot(ticks, idfScores, color = "b", label = "IDF Bayes")
plt.plot(ticks, scores, color = "r", label = "Bayes")

plt.xlabel('Size Difference')
plt.ylabel('Accuracy')
plt.title('Accuracy by Size Difference')

plt.legend()
 
plt.tight_layout()
plt.show()

#print("Accuracy of prediction is : ", model.score(XTest, YTest) * 100)
#print("Confussion Matrix:\n", confusion_matrix(YTest, predicted))

