import pandas as pd
import time
from os import listdir
from os.path import basename, join
from tqdm import tqdm
from xml.etree import ElementTree
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import re
import string
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import HashingVectorizer
from Generic import vectorize_text


def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print(
            [
                (vectorizer.get_feature_names()[i], topic[i])
                for i in topic.argsort()[: -top_n - 1 : -1]
            ]
        )


def main():
    add_stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    PATH = "./data/trainning/"
    xml_paths = []
    for pack in tqdm(listdir(PATH)):
        files = listdir(join(PATH, pack))
        for f in files:
            if ".nxml" in basename(f):
                xml_paths.append((join(PATH, pack, f), pack))
    dataset_abstracts = []
    abstracts = []

    for i, xml in enumerate(xml_paths):
        abstract = ""
        tree = ElementTree.parse(xml[0])
        root = tree.getroot()
        for p in root.findall("front/article-meta/abstract/sec/p"):
            abstract += " ".join(p.itertext())
    vectorize_text(
        "Esta mierda recibe un fucking texto, lo limpia, y lo vecotriza, ciao"
    )


if __name__ == "__main__":
    main()
