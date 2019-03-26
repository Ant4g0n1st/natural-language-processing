import nltk

tagged = nltk.corpus.cess_esp.tagged_words()
freq = nltk.FreqDist(tag for (w, tag) in tagged)
freq.plot(25)

