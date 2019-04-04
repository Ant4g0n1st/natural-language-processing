from bs4 import BeautifulSoup
import pickle
import nltk
import re

from gensim import corpora, models
import numpy as np

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

#name = input('Nombre del archivo : ')
#raw = ReadTextFromHTML(name)

sent_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')

word_regex = re.compile(r'^\w+[\n]*$')
digit_regex = re.compile(r'\d+')

tagger = LoadSpanishTagger()
stop = LoadStopwords()
stemMap = LoadStems()

f = open("e990101.htm", 'r')
text = f.read().lower().split("html>")
f.close()

chunks = []

for chunk in text:
    if chunk.strip().startswith("http"):
        continue
    soup = BeautifulSoup(chunk, "html.parser")
    tokens = [ w.lower() for w in nltk.word_tokenize(soup.get_text()) ]
    tokens = list(filter(lambda w : word_regex.match(w) != None, tokens))
    tokens = list(filter(lambda w : digit_regex.match(w) == None, tokens))
    tokens = list(filter(lambda w : w not in stop, tokens))
    tokens = [ stemMap.get(w, w) for w in tokens ]
    chunks.append(tokens)

dictionary = corpora.Dictionary(chunks)
print(dictionary)

corpus = [ dictionary.doc2bow(text) for text in chunks ]

# build tf-idf feature vectors
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
# fix the number of topics
total_topics = 2
# build the topic model
lsi = models.LsiModel(corpus_tfidf,
                      id2word=dictionary,
                      num_topics=total_topics)

for index, topic in lsi.print_topics(total_topics):
    print('Topic #'+str(index+1))
    print(topic)

tfidf = models.TfidfModel(mapped_corpus)
corpus_tfidf = tfidf[mapped_corpus]
lda = models.LdaModel(corpus_tfidf,
                      id2word=dictionary,
                      iterations=1000,
                      num_topics=total_topics)

"""
f = open("bla.txt", "w")
for bla in chunks:
    f.write("\n\n\n" + " ".join(bla))
f.close()
"""

