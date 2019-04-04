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

sent_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')

word_regex = re.compile(r'^\w+[\n]*$')
digit_regex = re.compile(r'\d+')

tagger = LoadSpanishTagger()
stop = LoadStopwords()
stemMap = LoadStems()

tokens = [ w.lower() for w in nltk.word_tokenize(raw) ]
tokens = list(filter(lambda w : word_regex.match(w) != None, tokens))
tokens = list(filter(lambda w : digit_regex.match(w) == None, tokens))
tokens = list(filter(lambda w : w not in stop, tokens))
tokens = [ stemMap.get(w, w) for w in tokens ]

vocabulary = sorted(set(tokens))
n = len(vocabulary)

context = [ [ 0 for y in range(0, n) ] for x in range(0, n) ]
contextCount = [ 0 for x in range(0, n) ]
p = [ 0 for x in range(0, n) ]
index = dict()
m = 0

for k in range(0, n):
    index[vocabulary[k]] = k

sentences = sent_tokenizer.tokenize(raw)
m = len(sentences)

for sentence in sentences: 

    tokens = list(map(lambda w : w.lower(), nltk.word_tokenize(sentence)))
    tokens = list(filter(lambda w : word_regex.match(w) != None, tokens))
    tokens = list(filter(lambda w : digit_regex.match(w) == None, tokens))
    tokens = list(filter(lambda w : w not in stop, tokens))
    tokens = set(stemMap.get(w, w) for w in tokens)

    indices = [ index[w] for w in tokens if w in index ]
    z = len(indices)

    for k in range(0, z):
        context[indices[k]][indices[k]] += 1
        contextCount[indices[k]] += 1
        p[indices[k]] += 1
        for l in range(k + 1, z):
            context[indices[k]][indices[l]] += 1
            context[indices[l]][indices[k]] += 1

idf = [ math.log((m + 1) / x) for x in contextCount ]

for k in range(0, n):
    p[k] = math.log1p(p[k]) * idf[k]

total = sum(p)

p = list(map(lambda x : x / total, p))

total = 0
for k in range(0, n):
    for l in range(0, n):
        context[k][l] = math.log1p(context[k][l]) * idf[l]
    total += sum(context[k])

for k in range(0, n):
    for l in range(0, n):
        context[k][l] /= total
        #context[k][l] *= idf[l] 
        #context[k][l] = math.log1p(context[k][l]) * idf[l]

while True:

    word = input('Ingrese la palabra : ')

    if word not in vocabulary:
        print("La palabra no se encuentra en el vocabulario.")
        continue

    x = index[word]
    h = []

    px, npx = p[x], 1 - p[x]
    for k in range(0, n):
        pk, npk = p[k], 1 - p[k]
        p11 = context[k][x]
        p10 = px - p11
        p01 = pk - p11
        p00 = p11 + p10 + p01
        p00 = 1 - p00
        h0 = 0
        if p00 > 0:
            h0 -= p00 * math.log2(p00)
        if p10 > 0:
            h0 -= p10 * math.log2(p10)
        h1 = 0
        if p01 > 0:
            h1 -= p01 * math.log2(p01)
        if p11 > 0:
            h1 -= p11 * math.log2(p11)
        h.append((h0 * npk + h1 * pk, k))

    h = sorted(h, reverse = True)

    f = open(word + "-entropy.txt", 'w')
    for (x, y) in h:
        f.write(vocabulary[y] + " \t " + str(x) + "\n")
    f.close()
    
    x = index[word]
    i = []

    px, npx = p[x], 1 - p[x]
    for k in range(0, n):
        pk, npk = p[k], 1 - p[k]
        p11 = context[k][x]
        p10 = px - p11
        p01 = pk - p11
        p00 = p11 + p10 + p01
        p00 = 1 - p00
        ik = 0
        if p00 > 0:
            ik += p00 * math.log2(p00 / (npx * npk))
        if p01 > 0:
            ik += p01 * math.log2(p01 / (npx * pk))
        if p10 > 0:
            ik += p10 * math.log2(p10 / (px * npk))
        if p11 > 0:
            ik += p11 * math.log2(p11 / (px * pk))
        i.append((ik, k))

    i = sorted(i, reverse = True)

    f = open(word + "-mi.txt", 'w')
    for (x, y) in i:
        f.write(vocabulary[y] + " \t " + str(x) + "\n")
    f.close()

    if input('¿Desea intentar de nuevo? [Y, N] : ').strip().lower() == "y":
        continue

    break

