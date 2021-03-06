import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle


def main():
    data_frame = pd.read_csv('dataframeVectors.csv')
    ids = data_frame["id"].to_list()
    no_ids = data_frame.loc[:, data_frame.columns != "id"]

    main_topics = []
    for i, row in no_ids.iterrows():
        topics = []
        for topic in row:
            lista = topic.replace('[', '').replace(
                ']', '').replace('\n', '').split(' ')
            filtered = [float(x) for x in lista if x != '']
            topics.append(filtered)
        main_topics.append(topics)

    main_topics = np.array(main_topics)
    nsamples, nx, ny = main_topics.shape
    main_topics = np.reshape(main_topics, (nsamples, nx*ny))
    k_means = pickle.load(open('model.pckl', 'rb'))
    # k_means = KMeans(n_clusters=5)
    # k_means.fit(main_topics)
    # pickle.dump(k_means, open('model.pckl', 'wb'))
    predicts = k_means.predict(main_topics)
    plt.hist(predicts)
    plt.show()
    df = DataFrame({'ids': ids, 'class': predicts})
    df.to_csv('cluster.csv', index=False)


if __name__ == "__main__":
    main()
