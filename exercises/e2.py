import nltk

patterns = []

for sentence in nltk.corpus.cess_esp.tagged_sents():
    for (w1, t1), (w2, t2), (w3, t3) in nltk.trigrams(sentence):
        if t1 == None or t3 == None:
            continue
        if t1.startswith('n') and w2.lower() == "de" and t3.startswith('n'):
            patterns.append((w1.lower(), w2.lower(), w3.lower()))

patterns = sorted(set(patterns), key = lambda x : len(x[0]), reverse = True)

for x in patterns:
    print(" ".join(x))

