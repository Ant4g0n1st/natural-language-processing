from sklearn.naive_bayes import MultinomialNB
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

#sent_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')

#word_regex = re.compile(r'^\w+[\n]*$')
#digit_regex = re.compile(r'\d+')

tagger = LoadSpanishTagger()
#stop = LoadStopwords()
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

docFreq = [ [ 0 for x in range(0, n) ] for y in range(0, m) ] 

for k in range(0, m):
    z = len(examples[k])
    for l in range(0, z):
        docFreq[k][index[examples[k][l]]] += 1

X = numpy.matrix(docFreq)
y = numpy.array(label)

try:
    f = open("model.pickle", "rb")
    #clf = pickle.load(f)
    f.close()
except:
    print("Training...")

    clf = MultinomialNB()
    clf.fit(X, y)

    f = open("model.pickle", "wb")
    pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
    f.close()

from sklearn.model_selection import train_test_split
XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size = 0.2)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 

clf = MultinomialNB()
clf.fit(XTrain, YTrain)

predicted = clf.predict(XTest)

print("Accuracy of prediction is : ", clf.score(XTest, YTest) * 100)
print("Confussion Matrix:\n", confusion_matrix(YTest, predicted))
print("Report:\n", classification_report(YTest, predicted))

