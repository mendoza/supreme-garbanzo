import pickle
import sys
from tools.Generic import vectorize_text
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from xml.etree import ElementTree


def getArguments():
    try:
        abstractPath = sys.argv[1]
        return abstractPath
    except IndexError:
        print("esta mal...")


def main():
    abstractPath = getArguments()
    with open(abstractPath, 'r') as abstractFile:
        abstract = abstractFile.read()

        # tree = ElementTree.parse(
        #     './data/training/PMC100324/1471-2407-2-3.nxml')
        # root = tree.getroot()
        # for p in root.findall("front/article-meta/abstract/sec/p"):
        #     abstract += " ".join(p.itertext())
        vector, _ = vectorize_text(abstract)
        vector = np.array(vector)
        nx, ny = vector.shape
        vector = vector.reshape(-1, nx*ny)
        kmeans = KMeans()
        kmeans = pickle.load(open('model.pckl', 'rb'))
        predict = kmeans.predict(vector)
        clusters = pd.read_csv('./cluster.csv')
        data = clusters.loc[clusters["class"] == predict[0], :]
        print(data["ids"].to_list())


if __name__ == "__main__":
    main()
