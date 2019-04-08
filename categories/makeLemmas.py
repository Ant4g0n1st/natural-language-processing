import pickle

f = open("lemma.en.txt", "r")
lemmas = dict()
for line in f:
    spl = line.strip().split(" ")
    lemma = spl[0].strip().split("/")[0]
    words = spl[-1].strip().split(",")
    for w in words:
        lemmas[w] = lemma
f.close()

f = open("lemmas.en.pickle", "wb")
pickle.dump(lemmas, f, pickle.HIGHEST_PROTOCOL)
f.close()

