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

def BM25(context):
    b = 0.75 # Experimental value. 
    k = 1.2 # Experimental value.
    n = len(context)

    avg = 0

    for x in range(0, n):
        avg = sum(context[x]) 

    avg = avg / n

    for x in range(0, n):
        s = sum(context[x])
        for y in range(0, n):
            v = (k + 1) * context[x][y]
            v = v / (context[x][y] + k * (1 - b + b * (s / avg)))
            context[x][y] = v
        s = sum(context[x])
        for y in range(0, n):
            context[x][y] = context[x][y] / s

    return context

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
contextCount = [0 for x in range(0, n)]
indices = [index[x] for x in tokens]

m = len(indices)
w = 4 # For window-8

for k in range(0, m):
    for l in range(1, w + 1):
        if k - l > 0:
            context[indices[k]][indices[k - l]] += 1
            contextCount[indices[k - l]] += 1
        if k + l < m:
            context[indices[k]][indices[k + l]] += 1
            contextCount[indices[k + l]] += 1

idf = [math.log((m + 1) / x) for x in contextCount]

try:
    f = open(name.split('.')[0] + '.pkl', 'rb')
    context = pickle.load(f)
    f.close()
except:
    f = open(name.split('.')[0] + '.pkl', 'wb')
    context = BM25(context)  
    pickle.dump(context, f)
    f.close()

while True:

    word = input('Ingrese la palabra : ')

    if word not in vocabulary:
        print("La palabra no se encuentra en el vocabulario.")
        continue
    
    p = []

    for k in nouns: 
        s, queryIndex = 0, index[word]
        for l in range(0, n):
            s += context[queryIndex][l] * context[k][l] * idf[l]
        p.append((s, k))

    p = sorted(p)
    p.reverse()

    f = open("similar-to-" + word + ".txt", 'w')
    for (x, y) in p:
        f.write(vocabulary[y] + " \t " + str(x) + "\n")
    f.close()

    if input('¿Desea intentar de nuevo? [Y, N] : ').strip().lower() == "y":
        continue

    break

