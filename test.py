import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import HashingVectorizer
import re
import string


def LDA(cv, abstract):

    data_cv = cv.fit_transform(abstract)
    data_stop = pd.DataFrame(
        data_cv.toarray(), columns=cv.get_feature_names())
    data_stop.to_csv("test.csv")

    # Build a Latent Dirichlet Allocation Model
    lda_model = LatentDirichletAllocation(
        n_components=5, max_iter=10, learning_method='online')
    lda_Z = lda_model.fit_transform(data_cv)

    return lda_model, lda_Z


def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
# Apply a second round of cleaning


def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text


def main():
    add_stop_words = stopwords.words('english')
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    data_frame = pd.read_csv('dataframeVectors.csv')
    no_ids = data_frame.loc[:, data_frame.columns != "ID"]
    # dataFrame = []

    main_topics = []
    for _, row in no_ids.iterrows():
        for topic in row:
            lista = topic.replace('[', '').replace(
                ']', '').replace('\n', '').split(' ')
            filtered = [float(x) for x in lista if x != '']
            main_topics.append(filtered)
        # dataFrame.append(topics)

    # topics = pd.DataFrame(dataFrame, columns=[
    #                     "Topic1", "Topic2", "Topic3", "Topic4", "Topic5"])

    data_frame = pd.read_csv('dataframeVectors.csv')
    no_ids = data_frame.loc[:, data_frame.columns != "ID"]
    # dataFrame = []

    topics = []
    for _, row in no_ids.iterrows():
        for topic in row:
            lista = topic.replace('[', '').replace(
                ']', '').replace('\n', '').split(' ')
            filtered = [float(x) for x in lista if x != '']
            topics.append(filtered)
        abstract = "Microarray classification poses many challenges for data analysis, given that a gene expression data set may consist of dozens of observations with thousands or even tens of thousands of genes. In this context, feature subset selection techniques can be very useful to reduce the representation space to one that is manageable by classification techniques. In this work we use the discretized multivariate joint entropy as the basis for a fast evaluation of gene relevance in a Microarray Gene Expression context. The proposed algorithm combines a simulated annealing schedule specially designed for feature subset selection with the incrementally computed joint entropy, reusing previous values to compute current feature subset relevance. This combination turns out to be a powerful tool when applied to the maximization of gene subset relevance. Our method delivers highly interpretable solutions that are more accurate than competing methods. The algorithm is fast, effective and has no critical parameters. The experimental results in several public-domain microarray data sets show a notoriously high classification performance and low size subsets, formed mostly by biologically meaningful genes. The technique is general and could be used in other similar scenarios."
        abstract = clean_text_round1(abstract)
        abstract = clean_text_round2(abstract)
        word_tokens = word_tokenize(abstract)
        filtered_sentence = [
            stemmer.stem(w) for w in word_tokens if not w.lower() in add_stop_words and not len(w) < 4]
        abstract = ' '.join(filtered_sentence)

    stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)
    cv = CountVectorizer(stop_words=stop_words)

    vectorizer = HashingVectorizer(n_features=50)
    abstract_model, abstract_Z = LDA(cv, [abstract])

    temas = []
    for idx, topic in enumerate(abstract_model.components_):
        nested_topics = []
        for i in (topic.argsort()[:-3 - 1:-1]):
            nested_topics.append(cv.get_feature_names()[i])
        temas.append(" ".join(nested_topics))
    vectorizer.fit_transform([temas[0]]).toarray()[0],
    
    main = [vectorizer.fit_transform([temas[1]]).toarray()[0], vectorizer.fit_transform([temas[2]]).toarray(
    )[0], vectorizer.fit_transform([temas[3]]).toarray()[0], vectorizer.fit_transform([temas[4]]).toarray()[0]]
    
    k_means = KMeans(n_clusters=5)
    k_means.fit(main_topics)
    predicts = k_means.predict(main)


if __name__ == "__main__":
    main()
