from bs4 import BeautifulSoup
import pickle
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

"""
name = input('Nombre del archivo : ')
raw = ReadTextFromHTML(name)

sent_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')

tagger = LoadSpanishTagger()

sentences = []
nouns = set()
tokens = []
"""

f = open('improved-spanish-tagger.pkl', 'rb')
tagger = pickle.load(f)
f.close()

word_regex = re.compile(r'^\w+[\n]*$')
digit_regex = re.compile(r'^\d+$')

#stop = LoadStopwords()
stemMap = LoadStems()

untagged = []

#for sentence in sent_tokenizer.tokenize(raw):
while True:
    
    sentence = input('Ingrese una oración: ')

    sentTokens = list(map(lambda x : x.lower(), nltk.word_tokenize(sentence)))

    sentTokens = list(filter(lambda x : word_regex.match(x) != None, sentTokens))

    sentTokens = list(filter(lambda x : digit_regex.match(x) == None, sentTokens))

    #sentTokens = list(filter(lambda x : x not in stop, sentTokens))

    sentTokens = [stemMap.get(x, x) for x in sentTokens]

    print(tagger.tag(sentTokens))

"""
    for (w, tag) in tagger.tag(sentTokens):
        if tag == None:
            untagged.append(w[::-1]);

print("\n".join(map(lambda x : x[::-1], sorted(set(untagged)))));
"""

