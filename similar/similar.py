from bs4 import BeautifulSoup

name = input('Nombre del archivo : ')

f = open(name)
raw = BeautifulSoup(f.read(), 'html.parser').get_text()
f.close()

import nltk
import re

word_regex = re.compile(r'^\w+[\n]*$')
digit_regex = re.compile(r'^\d+$')

tokens = list(map(lambda x : x.lower(), nltk.word_tokenize(raw)))

tokens = list(filter(lambda x : word_regex.match(x) != None, tokens))

tokens = list(filter(lambda x : digit_regex.match(x) == None, tokens))

f = open('stop.txt')
stop = set(f.read().split())
f.close()

tokens = list(filter(lambda x : x not in stop, tokens))

stemMap = dict()

f = open('stems.txt')
for line in f:  
    line = line.split()
    stemMap[line[0]] = line[-1]    
f.close()

tokens = [stemMap.get(x, x) for x in tokens]

f = open('processed.txt', 'w')
f.write(" ".join(tokens))
f.close()

text = nltk.Text(tokens)

freq = nltk.FreqDist(text)

f = open('frequency.txt', 'w')
for x, y in freq.most_common():
    f.write(F"{ x } { y }\n")
f.close()

vocabulary = sorted(set(tokens))
n = len(vocabulary)
index = dict()

for k in range(0, n):
    index[vocabulary[k]] = k

context = [[0 for y in range(0, n)] for x in range(0, n)]
indexes = [index[x] for x in tokens]
m = len(indexes)
w = 4 # For window-8

for k in range(0, m):
    for l in range(1, w + 1):
        if k - l > 0:
            context[indexes[k]][indexes[k - l]] += 1
        if k + l < m:
            context[indexes[k]][indexes[k + l]] += 1

import numpy

while True:

    word = input('Ingrese la palabra : ')

    if word not in vocabulary:
        print("La palabra no se encuentra en el vocabulario.")
        continue
    
    p = []

    for k in range(0, n):
        c = numpy.dot(context[index[word]], context[k])
        c = c / numpy.linalg.norm(context[index[word]])
        c = c / numpy.linalg.norm(context[k])
        p.append((c, k))

    p = sorted(p)
    p.reverse()

    f = open(F'similar-to-{ word }.txt', 'w')
    for (x, y) in p:
        f.write(F"{ vocabulary[y] } \t { x }\n")
    f.close()

    break

