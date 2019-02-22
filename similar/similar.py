from bs4 import BeautifulSoup
import nltk
import re

name = input('Nombre del archivo : ')

f = open(name)
raw = BeautifulSoup(f.read(), 'html.parser').get_text()
f.close()

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

freq.plot(25)
