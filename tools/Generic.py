import nltk
import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import HashingVectorizer


def LDA(cv, text):

    data_cv = cv.fit_transform(text)
    data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_stop.to_csv("test.csv")

    # Build a Latent Dirichlet Allocation Model
    lda_model = LatentDirichletAllocation(
        n_components=5, max_iter=10, learning_method="online"
    )
    lda_Z = lda_model.fit_transform(data_cv)

    return lda_model, lda_Z


def clean_text_round1(text):
    """Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers."""
    text = text.lower()
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\w*\d\w*", "", text)
    return text


def clean_text_round2(text):
    """Get rid of some additional punctuation and non-sensical text that was missed the first time around."""
    text = re.sub("[‘’“”…]", "", text)
    text = re.sub("\n", "", text)
    return text


def vectorize_text(Text):
    print("Original Text\n")
    print(Text)
    print("-" * 100)

    clean_text = []
    add_stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english", ignore_stopwords=True)

    Text = clean_text_round1(Text)
    Text = clean_text_round2(Text)

    word_tokens = word_tokenize(Text)
    filtered_tokens = [
        stemmer.stem(w)
        for w in word_tokens
        if not w.lower() in add_stop_words and not len(w) < 4
    ]

    clean_text.append(" ".join(filtered_tokens))

    print("Clean text\n")
    print(clean_text)
    print("-" * 80)

    stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)
    cv = CountVectorizer(stop_words=stop_words)

    vectorizer = HashingVectorizer(n_features=5)

    df_data = []
    df_vector_data = []

    Text_model, Text_Z = LDA(cv, clean_text)
    topics = []
    for idx, topic in enumerate(Text_model.components_):
        nested_topics = []
        for i in topic.argsort()[: -3 - 1 : -1]:
            nested_topics.append(cv.get_feature_names()[i])
        topics.append(" ".join(nested_topics))
        # print ("Topic ", idx, " ".join(cv.get_feature_names()[i] for i in topic.argsort()[:-3 - 1:-1]))
    df_vector_data.append(
        (
            vectorizer.fit_transform([topics[0]]).toarray()[0],
            vectorizer.fit_transform([topics[1]]).toarray()[0],
            vectorizer.fit_transform([topics[2]]).toarray()[0],
            vectorizer.fit_transform([topics[3]]).toarray()[0],
            vectorizer.fit_transform([topics[4]]).toarray()[0],
        )
    )

    df_data.append((topics[0], topics[1], topics[2], topics[3], topics[4]))

    df_vector = pd.DataFrame(
        df_vector_data, columns=["Topic1", "Topic2", "Topic3", "Topic4", "Topic5"]
    )
    # df_vector.to_csv("dataframeVectors.csv", index=False)
    print("Vectorized Topics\n")
    print(df_vector.head())
    print("-" * 80)
    df = pd.DataFrame(
        df_data, columns=["Topic1", "Topic2", "Topic3", "Topic4", "Topic5"]
    )
    # df.to_csv("dataframe.csv", index=False)
    print("Topics\n")
    print(df.head())
    print("-" * 80)

    return df_vector_data
