import os
import re
import pandas as pd
from gensim.models.nmf import Nmf
from gensim.models import CoherenceModel
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt
from pprint import pprint
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.probability import FreqDist
# from Chapter06.lda_topic_sklearn import stopwords, bbc_dataset, new_example
# from Chapter06.lda_topic_gensim import preprocess, test_new_example


curerntPath = os.path.dirname(os.path.abspath(__file__))
bbc_dataset = os.path.join(curerntPath, "bbc-text.csv")
stopwords_file_path = os.path.join(curerntPath, "stopwords.csv")
stemmer = SnowballStemmer('english')


def create_nmf_model(id_dict, corpus, num_topics):
    nmf_model = Nmf(corpus=corpus,
        id2word=id_dict,
        num_topics=num_topics,
        random_state=100,
        chunksize=100,
        passes=50)
    return nmf_model

def plot_coherence(id_dict, corpus, texts):
    num_topics_range = range(2, 10)
    coherences = []
    for num_topics in num_topics_range:
        nmf_model = create_nmf_model(id_dict, corpus,
                                    num_topics)
        coherence_model_nmf = CoherenceModel(model=nmf_model, texts=texts,
            dictionary=id_dict,
            coherence='c_v')
        coherences.append(coherence_model_nmf.get_coherence())
    plt.plot(num_topics_range, coherences,
    color='blue', marker='o', markersize=5)
    plt.title('Coherence as a function of number of topics')
    plt.xlabel('Number of topics')
    plt.ylabel('Coherence')
    plt.grid()
    plt.show()


stopwords_file_path = os.path.join(curerntPath, "stopwords.csv")
stemmer = SnowballStemmer('english')


def read_in_csv(filename):
    file = open(filename, "r", encoding="utf-8")
    text = file.read()
    text = text.replace("\n", " ")
    words = nltk.tokenize.word_tokenize(text)
    freq_dist = FreqDist(word.lower() for word in words)
    words_with_frequencies = [(word, freq_dist[word]) for word in freq_dist.keys()]
    sorted_words = sorted(words_with_frequencies, key=lambda tup: tup[1])
    length_cutoff = int(0.02*len(sorted_words))
    stopwords = [tuple[0] for tuple in sorted_words[-length_cutoff:]]
    return stopwords

def get_stopwords(path=stopwords_file_path):
    stopwords = read_in_csv(path)
    stopwords = [word[0] for word in stopwords]
    stemmed_stopwords = [stemmer.stem(word) for word in stopwords]
    stopwords = stopwords + stemmed_stopwords
    return stopwords


stopwords = get_stopwords(stopwords_file_path)

def clean_data(df):
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r'\d', '', x))
    return df


def preprocess(df):
    df = clean_data(df)
    df['text'] = df['text'].apply(lambda x: simple_preprocess(x, deacc=True))
    df['text'] = df['text'].apply(lambda x: [word for word in x if word not in stopwords])
    return df


df = pd.read_csv(bbc_dataset)
df = preprocess(df)

texts = df['text'].values
id_dict = corpora.Dictionary(texts)
corpus = [id_dict.doc2bow(text) for text in texts]

number_topics = 5
nmf_model = create_nmf_model(id_dict, corpus, number_topics)

pprint(nmf_model.print_topics())
plot_coherence(id_dict, corpus, texts)
