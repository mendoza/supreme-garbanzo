import pandas as pd
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


def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
               for i in topic.argsort()[:-top_n - 1:-1]])


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


def main():
    add_stop_words = stopwords.words('english')
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    PATH = './data/trainning/'
    xml_paths = []
    for pack in tqdm(listdir(PATH)):
        files = listdir(join(PATH, pack))
        for f in files:
            if '.nxml' in basename(f):
                xml_paths.append((join(PATH, pack, f), pack))
    dataset_abstracts = []
    abstracts = []

    for xml in xml_paths:
        tree = ElementTree.parse(xml[0])
        root = tree.getroot()
        abstract = ""
        for p in root.findall('front/article-meta/abstract/sec/p'):
            abstract += ' '.join(p.itertext())
        abstract = clean_text_round1(abstract)
        abstract = clean_text_round2(abstract)
        word_tokens = word_tokenize(abstract)
        filtered_sentence = [
            stemmer.stem(w) for w in word_tokens if not w.lower() in add_stop_words and not len(w) < 4]
        abstracts.append(' '.join(filtered_sentence))
        dataset_abstracts.append(filtered_sentence)

    stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)
    cv = CountVectorizer(stop_words=stop_words)

    vectorizer = HashingVectorizer(n_features=50)

    df_data = []
    df_vector_data = []
    contador = 0
    for abstract in tqdm(dataset_abstracts):
        abstract_model, abstract_Z = LDA(cv, abstracts)
        topics = []
        for idx, topic in enumerate(abstract_model.components_):
            nested_topics = []
            for i in (topic.argsort()[:-3 - 1:-1]):
                nested_topics.append(cv.get_feature_names()[i])
            topics.append(" ".join(nested_topics))
            # print ("Topic ", idx, " ".join(cv.get_feature_names()[i] for i in topic.argsort()[:-3 - 1:-1]))
        df_vector_data.append((xml_paths[contador][1], vectorizer.fit_transform([topics[0]]).toarray()[0],
                               vectorizer.fit_transform([topics[1]]).toarray()[0], vectorizer.fit_transform([topics[2]]).toarray()[0], vectorizer.fit_transform([topics[3]]).toarray()[0], vectorizer.fit_transform([topics[4]]).toarray()[0]))

        df_data.append((xml_paths[contador][1], topics[0],
                       topics[1], topics[2], topics[3], topics[4]))
        contador += 1

    df_vector = pd.DataFrame(df_vector_data, columns=[
                             'ID', 'Topic1', 'Topic2', 'Topic3', 'Topic4', 'Topic5'])
    df_vector.to_csv("dataframeVectors.csv", index=False)

    df = pd.DataFrame(df_data, columns=[
                      'ID', 'Topic1', 'Topic2', 'Topic3', 'Topic4', 'Topic5'])
    df.to_csv("dataframe.csv", index=False)

if __name__ == '__main__':
    main()
