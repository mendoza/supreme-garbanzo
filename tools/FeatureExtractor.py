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

    stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)
    cv = CountVectorizer(stop_words=stop_words)
    data_cv = cv.fit_transform(abstracts)
    data_stop = pd.DataFrame(
        data_cv.toarray(), columns=cv.get_feature_names())
    data_stop.to_csv("test.csv")

    # Build a Latent Dirichlet Allocation Model
    lda_model = LatentDirichletAllocation(
        n_components=5, max_iter=10, learning_method='online')
    lda_Z = lda_model.fit_transform(data_cv)
    print(lda_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
    print("LDA Model:")
    print_topics(lda_model, cv)
    print("=" * 20)
    # Let's see how the first document in the corpus looks like in different topic spaces
    print(lda_Z[0])


if __name__ == '__main__':
    main()
