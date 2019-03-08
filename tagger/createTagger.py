import pickle
import nltk
import re

sents = nltk.corpus.cess_esp.tagged_sents()
#tagger = nltk.UnigramTagger(train = sents, backoff = nltk.DefaultTagger('AHOCORASICK'))

patterns = [
    (r'.*ístas$'    ,   'nccps00'),
    (r'.*istas$'    ,   'nccps00'),
    (r'.*selo$'     ,   'vmg0000')
]

tagger = nltk.UnigramTagger(train = sents, backoff = nltk.RegexpTagger(patterns))

f = open('improved-spanish-tagger.pkl', 'wb')
pickle.dump(tagger, f, pickle.HIGHEST_PROTOCOL)
f.close()

