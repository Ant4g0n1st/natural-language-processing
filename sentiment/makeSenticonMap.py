from bs4 import BeautifulSoup as Soup
import pickle

fileName = "senticon.es.xml"

pol = dict()

with open(fileName, "r") as f:
    soup = Soup(f.read(), "xml")
    for lemma in soup.find_all("lemma"):
        pol[(lemma.get_text().strip(), lemma["pos"])] = float(lemma["pol"])

print(pol)

with open("polarity.pkl", "wb") as f:
    pickle.dump(pol, f, pickle.HIGHEST_PROTOCOL)

