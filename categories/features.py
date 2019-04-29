import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import numpy as np
import pickle
import random
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

# Print Confusion Matrix.

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# Functions End 

stemMap = LoadStems()

examples = []
label = []

f = open("spam.txt", "r", encoding = "latin-1")
for line in f:
    tokens = nltk.word_tokenize(line.strip())
    if len(tokens) == 0:
        continue
    examples.append(" ".join(tokens[ : -2]))
    if tokens[-1] == "spam":
        label.append(0) 
    else: 
        label.append(1)
f.close()

m = len(examples)

def FeatureExtraction(text):
    nonDig = re.compile("\D+") 
    return { "digit-count" : len(nonDig.sub("", text)) } 

features = [ ( FeatureExtraction(examples[k]), label[k] ) for k in range(0, m) ]
random.shuffle(features)

testSize = int(len(features) * 0.2)
train, test = features[testSize : ], features[ : testSize]
classifier = nltk.NaiveBayesClassifier.train(train)

pred = [ classifier.classify(featureSet) for (featureSet, label) in test ]
testLabels = [ label for (featureSet, label) in test ]

confusion = nltk.ConfusionMatrix(testLabels, pred)
cm = np.matrix(confusion._confusion)

plot_confusion_matrix(cm, classes = [ "spam", "ham" ], title = "Spam/Ham Confusion Matrix")

plt.show()

print(nltk.classify.accuracy(classifier, test))
