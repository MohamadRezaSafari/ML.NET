import os
import re
import pandas as pd
from gensim.models.ldamodel import LdaModel
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt
from pprint import pprint
import nltk
from nltk.probability import FreqDist
from nltk.stem.snowball import SnowballStemmer

# from Chapter06.lda_topic import stopwords, bbc_dataset, clean_data


curerntPath = os.path.dirname(os.path.abspath(__file__))
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
bbc_dataset = os.path.join(curerntPath, "bbc-text.csv")

def clean_data(df):
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r'\d', '', x))
    return df


def preprocess(df):
    df = clean_data(df)
    df['text'] = df['text'].apply(lambda x: simple_preprocess(x, deacc=True))
    df['text'] = df['text'].apply(lambda x: [word for word in x if word not in stopwords])
    return df

def create_lda_model(id_dict, corpus, num_topics):
    lda_model = LdaModel(corpus=corpus,
        id2word=id_dict,
        num_topics=num_topics,
        random_state=100,
        chunksize=100,
        passes=10)
    return lda_model


df = pd.read_csv(bbc_dataset)
df = preprocess(df)


texts = df['text'].values
id_dict = corpora.Dictionary(texts)
corpus = [id_dict.doc2bow(text) for text in texts]

number_topics = 5
lda_model = create_lda_model(id_dict, corpus, number_topics)

pprint(lda_model.print_topics())



new_example = """Manchester United players slumped to the
turf
at full-time in Germany on Tuesday in acknowledgement of
what their
latest pedestrian first-half display had cost them. The
3-2 loss at
RB Leipzig means United will not be one of the 16 teams
in the draw
for the knockout stages of the Champions League. And this
is not the
only price for failure. The damage will be felt in the
accounts, in
the dealings they have with current and potentially
future players
and in the faith the fans have placed in manager Ole
Gunnar Solskjaer.
With Paul Pogba's agent angling for a move for his client
and ex-United
defender Phil Neville speaking of a "witchhunt" against
his former team-mate
Solskjaer, BBC Sport looks at the ramifications and
reaction to a big loss for United."""


def save_model(lda, lda_path, id_dict, dict_path):
    lda.save(lda_path)
    id_dict.save(dict_path)


def load_model(lda_path, dict_path):
    lda = LdaModel.load(lda_path)
    id_dict = corpora.Dictionary.load(dict_path)
    return (lda, id_dict)


def test_new_example(lda, id_dict, input_string):
    input_list = clean_text(input_string)
    bow = id_dict.doc2bow(input_list)
    topics = lda[bow]
    print(topics)
    return topics


save_model(lda_model, model_path, id_dict, dict_path)
test_new_example(lda_model, id_dict, new_example)
