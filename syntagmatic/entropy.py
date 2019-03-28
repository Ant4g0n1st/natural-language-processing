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

#sent_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')

word_regex = re.compile(r'^\w+[\n]*$')
digit_regex = re.compile(r'\d+')

#tagger = LoadSpanishTagger()

#sentences = []
tokens = nltk.word_tokenize(raw)
nouns = set()

#sentTokens = list(map(lambda x : x.lower(), nltk.word_tokenize(sentence)))

tokens = list(filter(lambda x : word_regex.match(x) != None, tokens))

tokens = list(filter(lambda x : digit_regex.match(x) == None, tokens))

tokens = list(filter(lambda x : x not in stop, tokens))

tokens = [stemMap.get(x, x) for x in tokens]

vocabulary = sorted(set(tokens))
n = len(vocabulary)

index = dict()

for k in range(0, n):
    index[vocabulary[k]] = k

context = [[0 for y in range(0, n)] for x in range(0, n)]

#nouns = sorted([index[x] for x in nouns])
contextCount = [0 for x in range(0, n)]
indices = [index[x] for x in tokens]
p = [0 for x in range(0, n)]

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
    p[indices[k]] += 1

#idf = [math.log((m + 1) / x) for x in contextCount]

p = list(map(lambda x : x / m, p))

for k in range(0, n):
    s = sum(context[k]) 
    for l in range(0, n):
        context[k][l] = context[k][l] / s
        #context[k][l] = math.log1p(context[k][l]) * idf[l]

while True:

    word = input('Ingrese la palabra : ')

    if word not in vocabulary:
        print("La palabra no se encuentra en el vocabulario.")
        continue

    x = index[word]
    h = []

    for k in range(0, n):
        px, npx = p[x], 1 - p[x]
        pk, npk = p[k], 1 - p[k]
        p11 = context[x][k]
        p10 = pk - p11
        p01 = px - p11
        p00 = npk - p01 
        h0 = 0
        if p00 > 0:
            h0 -= p00 * math.log2(p00)
        if p10 > 0:
            h0 -= p10 * math.log2(p10)
        h0 *= npx
        h1 = 0
        if p01 > 0:
            h1 -= p01 * math.log2(p01)
        if p11 > 0:
            h1 -= p11 * math.log2(p11)
        h1 *= px
        h.append((h0 + h1, k))

    h = sorted(h)

    f = open("syntagmatic-to-" + word + ".txt", 'w')
    for (x, y) in h:
        f.write(vocabulary[y] + " \t " + str(x) + "\n")
    f.close()

    if input('¿Desea intentar de nuevo? [Y, N] : ').strip().lower() == "y":
        continue

    break

