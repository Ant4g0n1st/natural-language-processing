from bs4 import BeautifulSoup
import numpy as np
import pickle
import nltk
import re

# Functions Begin

# Read only the text from an HTML file.
def ReadHTMLFromFile(fileName):
    f = open(name)
    html = f.read()
    f.close()

    return html 

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
    f = open('lemmas.pkl', 'rb')
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

def Normalize(text):
    global d_regex
    global w_regex
    #global stop

    tokens = [ w.lower() for w in nltk.word_tokenize(text) ]
    tokens = list(filter(lambda w : w_regex.match(w) != None, tokens))
    tokens = list(filter(lambda w : d_regex.match(w) == None, tokens))
    #tokens = list(filter(lambda w : w not in stop, tokens))

    return tokens

# Functions End 

name = input('Nombre del archivo : ').strip()
raw = ReadHTMLFromFile(name)

sent_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')

w_regex = re.compile(r'^\w+[\n]*$')
d_regex = re.compile(r'\d+')

text = raw.lower().split("html>")

tagger = LoadSpanishTagger()
stop = LoadStopwords()
stemMap = LoadStems()

articles = []
nouns = set()

for article in text:
    article = article.strip()
    if article.startswith("http"):
        continue
    soup = BeautifulSoup(article, "html.parser")
    content = soup.get_text()
    tokens = []
    for sentence in sent_tokenizer.tokenize(content):
        tagged = tagger.tag(Normalize(sentence))
        for (w, tag) in tagged:
            if w in stop:
                continue
            if tag == None:
                tokens.append(w)
                continue
            tag = tag[0].lower()
            w = stemMap.get((w, tag), w)
            if tag == "n":
                nouns.add(w)
            tokens.append(w)
    articles.append(tokens)

nouns = sorted(nouns)
index = dict()
m = 0 # Number of nouns

for noun in nouns:
    index[noun] = m
    m += 1

nounFreq = [ 0 for x in range(0, m) ]

for article in articles:
    for w in article:
        k = index.get(w, -1)
        if k < 0:
            continue
        nounFreq[k] += 1

freqPair = [ (nounFreq[k], k) for k in range(0, m) ] 
freqPair = sorted(freqPair, reverse = True)

name = name.split(".")[0]
f = open(name + "-nouns.txt", "w")
for (x, k) in freqPair:
    f.write(nouns[k] + " " + str(x) + "\n")
f.close()

freq = np.array(nounFreq)
a = len(articles) 
bm25K = 1.2

bm25 = np.divide(freq * (bm25K + 1), freq + bm25K)

idf = np.log((a + 1) / freq)

bm25IDF = np.multiply(bm25, idf)

freqPair = [ (bm25IDF[k], k) for k in range(0, m) ]
freqPair = sorted(freqPair, reverse = True)

f = open(name + "-bm25-nouns.txt", "w")
for (x, k) in freqPair:
    f.write(nouns[k] + " " + str(x) + "\n")
f.close()

