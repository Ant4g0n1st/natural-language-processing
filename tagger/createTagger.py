import pickle
import nltk
import re

sents = nltk.corpus.cess_esp.tagged_sents()
#tagger = nltk.UnigramTagger(train = sents, backoff = nltk.DefaultTagger('AHOCORASICK'))

patterns = [
    (r'.*aca$'      ,   'n000000'),
    (r'.*ad$'       ,   'n000000'),
    (r'.*ada$'      ,   'a000000'),
    (r'.*c$'        ,   'n000000'),
    (r'.*ca$'       ,   'n000000'),
    (r'.*cha$'      ,   'n000000'),
    (r'.*deuda$'    ,   'n000000'),
    (r'.*ea$'       ,   'n000000'),
    (r'.*ga$'       ,   'n000000'),
    (r'.*ia$'       ,   'n000000'),
    (r'.*id$'       ,   'n000000'),
    (r'.*ida$'      ,   'a000000'),
    (r'.*ica$'      ,   'n000000'),
    (r'.*ía$'       ,   'n000000'),
    (r'.*ístas$'    ,   'n000000'),
    (r'.*istas$'    ,   'n000000'),
    (r'.*la$'       ,   'n000000'),
    (r'.*ld$'       ,   'n000000'),
    (r'.*ma$'       ,   'n000000'),
    (r'.*na$'       ,   'n000000'),
    (r'.*nd$'       ,   'n000000'),
    (r'.*nda$'      ,   'n000000'),
    (r'.*ña$'       ,   'n000000'),
    (r'.*oa$'       ,   'n000000'),
    (r'.*pa$'       ,   'n000000'),
    (r'.*ra$'       ,   'n000000'),
    (r'.*rd$'       ,   'n000000'),
    (r'.*rda$'      ,   'n000000'),
    (r'.*sa$'       ,   'n000000'),
    (r'.*suda$'     ,   'a000000'),
    (r'.*ta$'       ,   'n000000'),
    (r'.*tha$'      ,   'np0ss00'),
    (r'.*ud$'       ,   'n000000'),
    (r'.*va$'       ,   'n000000'),
    (r'.*ya$'       ,   'n000000'),
    (r'.*za$'       ,   'n000000')
]

tagger = nltk.UnigramTagger(train = sents, backoff = nltk.RegexpTagger(patterns))

f = open('improved-spanish-tagger.pkl', 'wb')
pickle.dump(tagger, f, pickle.HIGHEST_PROTOCOL)
f.close()

