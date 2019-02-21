from bs4 import BeautifulSoup
import nltk
import re

name = input('Nombre del archivo : ')

raw = BeautifulSoup(open(name).read(), 'html.parser').get_text()

word_regex = re.compile(r'^\w+[\n]*$')
digit_regex = re.compile(r'^\d+$')

tokens = list(map(lambda x : x.lower(), nltk.word_tokenize(raw)))

tokens = list(filter(lambda x : word_regex.match(x) != None, tokens))

tokens = list(filter(lambda x : digit_regex.match(x) == None, tokens))

print(tokens)
