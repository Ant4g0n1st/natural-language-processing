from bs4 import BeautifulSoup
import pickle
import numpy
import math
import nltk
import re

# Functions Begin

# Read only the text from an HTML file.
def ReadTextFromHTML(fileName):
    f = open(name)
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

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

names = [ "spam", "ham" ]

def GetValues(X, y, model, name = "spam"):  
    XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size = 0.2)
    model.fit(XTrain, YTrain)
    predicted = model.predict(XTest)

    return classification_report(YTest, predicted, target_names = names, output_dict = True)[name]

#print("Accuracy of prediction is : ", model.score(XTest, YTest) * 100)
#print("Confussion Matrix:\n", confusion_matrix(YTest, predicted))

idfX = numpy.array(idfFreq)
X = numpy.matrix(freq)
y = numpy.array(label)

values = [
    GetValues(X, y, MultinomialNB()),
    GetValues(idfX, y, MultinomialNB()),
    GetValues(X, y, LogisticRegression()),
    GetValues(idfX, y, LogisticRegression()),
    GetValues(X, y, LinearSVC()),
    GetValues(idfX, y, LinearSVC()),
    GetValues(X, y, KNeighborsClassifier()),
    GetValues(idfX, y, KNeighborsClassifier())
]

# Plotting the Values
import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
nMethods = 8
 
# create plot
fig, ax = plt.subplots()
index = np.arange(nMethods)
opacity = 0.8
width = 0.25

precision = plt.bar(index, [ values[x]["precision"] * 100 for x in range(0, nMethods) ],
    width, alpha = opacity, color = "r", label = "Precision") 

recall = plt.bar(index + width, [ values[x]["recall"] * 100 for x in range(0, nMethods) ],
    width, alpha = opacity, color = "g", label = "Recall") 

f1Score = plt.bar(index + 2 * width, [ values[x]["f1-score"] * 100 for x in range(0, nMethods) ],
    width, alpha = opacity, color = "b", label = "F1-Score") 
 
plt.xlabel('Method')
plt.ylabel('Scores')
plt.title('Scores by Method')

ticks = [ 
    "Bayes", 
    "IDF Bayes", 
    "Regr.", 
    "IDF Regr.", 
    "SVC", 
    "IDF SVC", 
    "KNN", 
    "IDF KNN"
]
plt.xticks(index + width, ticks) 
plt.legend()
 
plt.tight_layout()
plt.show()
