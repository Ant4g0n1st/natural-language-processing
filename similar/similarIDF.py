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
    f = open('stems.txt')
    stemMap = dict()

    for line in f:  
        line = line.split()
        stemMap[line[0]] = line[-1]    
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

name = input('Nombre del archivo : ')
raw = ReadTextFromHTML(name)

stop = LoadStopwords()
stemMap = LoadStems()

# To train the spanish POS tagger.
# tagger = nltk.UnigramTagger(nltk.corpus.cess_esp.tagged_sents())

sent_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')

word_regex = re.compile(r'^\w+[\n]*$')
digit_regex = re.compile(r'^\d+$')

tagger = LoadSpanishTagger()

sentences = []
nouns = set()
tokens = []

for sentence in sent_tokenizer.tokenize(raw):

    sentTokens = list(map(lambda x : x.lower(), nltk.word_tokenize(sentence)))

    sentTokens = list(filter(lambda x : word_regex.match(x) != None, sentTokens))

    sentTokens = list(filter(lambda x : digit_regex.match(x) == None, sentTokens))

    sentTokens = list(filter(lambda x : x not in stop, sentTokens))

    sentTokens = [stemMap.get(x, x) for x in sentTokens]

    tagged = tagger.tag(sentTokens)
    sentences.append(sentTokens)
    
    for k in range(0, len(tagged)):
        tokens.append(sentTokens[k])
        (word, tag) = tagged[k]
        if tag == None:
            continue
        if tag.startswith('n'):
            nouns.add(sentTokens[k])

vocabulary = sorted(set(tokens))
n = len(vocabulary)

index = dict()

for k in range(0, n):
    index[vocabulary[k]] = k

context = [[0 for y in range(0, n)] for x in range(0, n)]

nouns = sorted([index[x] for x in nouns])
indices = [index[x] for x in tokens]

m = len(indices)
w = 4 # For window-8

for k in range(0, m):
    for l in range(1, w + 1):
        if k - l > 0:
            context[indices[k]][indices[k - l]] += 1
        if k + l < m:
            context[indices[k]][indices[k + l]] += 1

valid = [False for x in range(0, n)]
for x in nouns:
    valid[x] = True

for k in range(0, n):
    for l in range(0, n):
        if not valid[l]:
            context[k][l] = 0
        context[k][l] = math.log1p(context[k][l])

while True:

    word = input('Ingrese la palabra : ')

    if word not in vocabulary:
        print("La palabra no se encuentra en el vocabulario.")
        continue
    
    p = []

    for k in range(0, n):
        c = numpy.dot(context[index[word]], context[k])
        d = numpy.linalg.norm(context[index[word]])
        if d > 0:
            c = c / d
        d = numpy.linalg.norm(context[k])
        if d > 0:
            c = c / d
        p.append((c, k))

    p = sorted(p)
    p.reverse()

    f = open("similar-to-" + word + ".txt", 'w')
    for (x, y) in p:
        f.write(vocabulary[y] + " \t " + str(x) + "\n")
    f.close()

    if input('¿Desea intentar de nuevo? [Y, N] : ').strip().lower() == "y":
        continue

    break

