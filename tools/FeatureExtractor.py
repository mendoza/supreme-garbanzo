import pandas as pd
from os import listdir
from os.path import basename, join
import numpy as np
from tqdm import tqdm
from xml.etree import ElementTree
from Generic import vectorize_text


def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print(
            [
                (vectorizer.get_feature_names()[i], topic[i])
                for i in topic.argsort()[: -top_n - 1: -1]
            ]
        )


def main():
    PATH = "./data/training/"
    xml_paths = []
    for pack in tqdm(listdir(PATH)):
        files = listdir(join(PATH, pack))
        for f in files:
            if ".nxml" in basename(f):
                xml_paths.append((join(PATH, pack, f), pack))
    ids = []
    vectors = []
    df = pd.DataFrame(
        {}, columns=["id", "Topic1", "Topic2", "Topic3", "Topic4", "Topic5"]
    )
    for i, xml in enumerate(tqdm(xml_paths)):
        abstract = ""
        tree = ElementTree.parse(xml[0])
        root = tree.getroot()
        for p in root.findall("front/article-meta/abstract/sec/p"):
            abstract += " ".join(p.itertext())

        if abstract != '':
            vector, word_presentation = vectorize_text(abstract)
            vectors.append(vector)
            ids.append(xml_paths[i][1])
            word_presentation["id"] = ids[i]
            df = df.append(word_presentation)

    df_vectors = pd.DataFrame({"id": [], "Topic1": [], "Topic2": [], "Topic3": [], "Topic4": [
    ], "Topic5": []}, columns=["Topic1", "Topic2", "Topic3", "Topic4", "Topic5"])
    for i, vector in enumerate(vectors):
        df_vectors = df_vectors.append({"id": ids[i],
                                        "Topic1": vector[0], "Topic2": vector[1], "Topic3": vector[2], "Topic4": vector[3], "Topic5": vector[4]}, ignore_index=True)
    df_vectors.to_csv("dataframeVectors.csv", index=False)
    df.to_csv("dataframe.csv", index=False)


if __name__ == "__main__":
    main()
