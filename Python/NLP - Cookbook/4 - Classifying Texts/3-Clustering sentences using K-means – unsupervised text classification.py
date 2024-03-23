import csv
import os
import nltk
import re
import string
import pandas as pd
from sklearn.cluster import KMeans
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.probability import FreqDist
from sklearn.model_selection import train_test_split
import pickle
# from Chapter01.tokenization import tokenize_nltk
# from Chapter01.dividing_into_sentences import divide_into_sentences_nltk
# from Chapter04.preprocess_bbc_dataset import get_data
# from Chapter04.keyword_classification import divide_data
# from Chapter04.preprocess_bbc_dataset import get_stopwords


curerntPath = os.path.dirname(os.path.abspath(__file__))
bbc_dataset = os.path.join(curerntPath, "bbc-text.csv")
stopwords_file_path = os.path.join(curerntPath, "stopwords.csv")
stemmer = SnowballStemmer('english')

def read_in_csv(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        data_read = [row for row in reader]
    return data_read


def get_stopwords(path=stopwords_file_path):
    stopwords = read_in_csv(path)
    stopwords = [word[0] for word in stopwords]
    stemmed_stopwords = [stemmer.stem(word) for word in stopwords]
    stopwords = stopwords + stemmed_stopwords
    return stopwords

stopwords = get_stopwords(stopwords_file_path)


def get_data(filename):
    data = read_in_csv(filename)
    data_dict = {}
    for row in data[1:]:
        category = row[0]
        text = row[1]
        if (category not in data_dict.keys()):
            data_dict[category] = []
        data_dict[category].append(text)
    for topic in data_dict.keys():
        print(topic, "\t", len(data_dict[topic]))
    return data_dict


def divide_data(data_dict):
    train_dict = {}
    test_dict = {}
    for topic in data_dict.keys():
        text_list = data_dict[topic]
        x_train, x_test = \
        train_test_split(text_list, test_size=0.2)
        train_dict[topic] = x_train
        test_dict[topic] = x_test
    return (train_dict, test_dict)

data_dict = get_data(bbc_dataset)
(train_dict, test_dict) = divide_data(data_dict)


all_training = []
all_test = []
for topic in train_dict.keys():
    all_training = all_training + train_dict[topic]
for topic in test_dict.keys():
    all_test = all_test + test_dict[topic]


def tokenize_and_stem(sentence):
    tokens = nltk.word_tokenize(sentence)
    filtered_tokens = [t for t in tokens if t not in
    stopwords and t not in
    string.punctuation and
    re.search('[a-zA-Z]', t)]
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def create_vectorizer(data):
    vec = TfidfVectorizer(max_df=0.90, max_features=200000, min_df=0.05, stop_words=stopwords, use_idf=True,
                            tokenizer=tokenize_and_stem,
                            ngram_range=(1,3))
    vec.fit(data)
    return vec



vectorizer = create_vectorizer(all_training)
matrix = vectorizer.transform(all_training)

km = KMeans(n_clusters=5, init='k-means++', random_state=0)
km.fit(matrix)


def make_predictions(test_data, vectorizer, km):
    predicted_data = {}
    for topic in test_data.keys():
        this_topic_list = test_data[topic]
        if (topic not in predicted_data.keys()):
            predicted_data[topic] = {}
        for text in this_topic_list:
            prediction = km.predict(vectorizer.transform([text]))[0]
            if (prediction not in predicted_data[topic].keys()):
                predicted_data[topic][prediction] = []
            predicted_data[topic][prediction].append(text)
    return predicted_data


def print_report(predicted_data):
    for topic in predicted_data.keys():
        print(topic)
        for prediction in predicted_data[topic].keys():
            print("Cluster number: ", prediction,"number of items: ", len(predicted_data[topic][prediction]))


predicted_data = make_predictions(test_dict, vectorizer, km)
print_report(predicted_data)
      

def print_most_common_words_by_cluster(all_training, km, num_clusters):
    clusters = km.labels_.tolist()
    docs = {'text': all_training, 'cluster': clusters}
    frame = pd.DataFrame(docs, index = [clusters])
    for cluster in range(0, num_clusters):
        this_cluster_text = frame[frame['cluster'] == cluster]
        all_text = " ".join(this_cluster_text['text'].astype(str))
        top_200 = get_most_frequent_words(all_text)
        print(cluster)
        print(top_200)
    return frame

print_most_common_words_by_cluster(all_training, km)


pickle.dump(km, open("bbc_kmeans.pkl", "wb"))
km = pickle.load(open("bbc_kmeans.pkl", "rb"))
