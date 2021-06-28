import pickle
import sys
from tools.Generic import vectorize_text
import numpy as np
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
        vector = np.array(vectorize_text(abstract))
        nx, ny = vector.shape
        vector = vector.reshape(-1, nx*ny)
        kmeans = KMeans()
        kmeans = pickle.load(open('model.pckl', 'rb'))
        print(kmeans.predict(vector))


if __name__ == "__main__":
    main()
