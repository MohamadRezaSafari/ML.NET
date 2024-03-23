import numpy as np
import pandas as pd
import string
import pickle
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from Chapter01.tokenization import tokenize_nltk
from Chapter04.unsupervised_text_classification import tokenize_and_stem
from Chapter04.preprocess_bbc_dataset import get_data
from Chapter04.keyword_classification import get_labels
from Chapter04.preprocess_bbc_dataset import get_stopwords



bbc_dataset = "Chapter04/bbc-text.csv"
stopwords_file_path = "Chapter01/stopwords.csv"
stopwords = get_stopwords(stopwords_file_path)

data_dict = get_data(bbc_dataset)
le = get_labels(list(data_dict.keys()))


def create_dataset(data_dict, le):
    text = []
    labels = []
    for topic in data_dict:
        label = le.transform([topic])
        text = text + data_dict[topic]
        this_topic_labels = [label[0]]*len(data_dict[topic])
        labels = labels + this_topic_labels
    docs = {'text':text, 'label':labels}
    frame = pd.DataFrame(docs)
    return frame


def split_dataset(df, train_column_name,
    gold_column_name, test_percent):
    X_train, X_test, y_train, y_test = train_test_split(df[train_column_name],
                df[gold_column_name],
                test_size=test_percent,
                random_state=0)
    return (X_train, X_test, y_train, y_test)



def create_and_fit_vectorizer(training_text):
    vec = TfidfVectorizer(max_df=0.90, min_df=0.05,
                        stop_words=stopwords, use_idf=True,
                        tokenizer=tokenize_and_stem,
                        ngram_range=(1,3))
    return vec.fit(training_text)



df = create_dataset(data_dict, le)
(X_train, X_test, y_train, y_test) = split_dataset(df, 'text', 'label')
vectorizer = create_and_fit_vectorizer(X_train)
X_train = vectorizer.transform(X_train).todense()
X_test = vectorizer.transform(X_test).todense()


def train_svm_classifier(X_train, y_train):
    clf = svm.SVC(C=1, kernel='linear',
    decision_function_shape='ovo')
    clf = clf.fit(X_train, y_train)
    return clf


def evaluate(clf, X_test, y_test, le):
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred,
            labels=le.transform(le.classes_),
            target_names=le.classes_))
    

clf = train_svm_classifier(X_train, y_train)
pickle.dump(clf, open("Chapter04/bbc_svm.pkl", "wb"))
evaluate(clf, X_test, y_test, le)