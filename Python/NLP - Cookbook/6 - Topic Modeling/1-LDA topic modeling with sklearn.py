import os
import re
import string
import nltk
from nltk.probability import FreqDist
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
# from Chapter04.preprocess_bbc_dataset import get_stopwords
# from Chapter04.unsupervised_text_classification import tokenize_and_stem

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


def tokenize_and_stem(sentence):
    tokens = nltk.word_tokenize(sentence)
    filtered_tokens = [t for t in tokens if t not in string.punctuation]
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def create_count_vectorizer(documents):
    count_vectorizer = CountVectorizer(stop_words=stopwords,
    tokenizer=tokenize_and_stem)
    data = count_vectorizer.fit_transform(documents)
    return (count_vectorizer, data)


def create_count_vectorizer(documents):
    count_vectorizer = CountVectorizer(stop_words=stopwords,
    tokenizer=tokenize_and_stem)
    data = count_vectorizer.fit_transform(documents)
    return (count_vectorizer, data)


def clean_data(df):
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r'\d', '', x))
    return df


def create_and_fit_lda(data, num_topics):
    lda = LDA(n_components=num_topics, n_jobs=-1)
    lda.fit(data)
    return lda


def get_most_common_words_for_topics(model, vectorizer, n_top_words):
    words = vectorizer.get_feature_names_out()
    word_dict = {}
    for topic_index, topic in enumerate(model.components_):
        this_topic_words = [words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        word_dict[topic_index] = this_topic_words
    return word_dict
    

def print_topic_words(word_dict):
    for key in word_dict.keys():
        print(f"Topic {key}")
        print("\t", word_dict[key])



df = pd.read_csv(bbc_dataset)
df = clean_data(df)
documents = df['text']
number_topics = 5
(vectorizer, data) = create_count_vectorizer(documents)
lda = create_and_fit_lda(data, number_topics)

topic_words = get_most_common_words_for_topics(lda, vectorizer, 10)
print_topic_words(topic_words)
