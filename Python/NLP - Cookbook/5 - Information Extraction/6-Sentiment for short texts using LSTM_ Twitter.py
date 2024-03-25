import re
import pandas as pd
from tqdm import tqdm
from wordseg import segment
import html
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Densefrom 
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
# from Chapter04.lstm_classification import plot_model



MAX_NUM_WORDS = 50000
EMBEDDING_DIM = 500
twitter_csv = "Chapter05/training.1600000.processed.noemoticon.csv"
english_twitter = "Chapter05/twitter_english.csv"
tqdm.pandas()


def filter_english(df, save_path):
    df['language'] = df['tweet'].progress_apply(lambda t: lang_detect(t))
    df = df[df['language'] == 'en']
    df.to_csv(save_path, encoding="latin1")
    return df

def get_data(filename, save_path,
    num_datapoints=80000):
    df = pd.read_csv(filename, encoding="latin1")
    df.columns = ['sentiment', 'id', 'date', 'query',
    'username', 'tweet']
    df = pd.concat([df.head(num_datapoints),
    df.tail(num_datapoints)])
    df = filter_english(df, save_path)
    return df


def clean_data(df):
    #Lowercase all tweets
    df['tweet'] = df['tweet'].progress_apply(lambda t: t.lower())
    #Decode HTML
    df['tweet'] = df['tweet'].progress_apply(lambda t: html.unescape(t))
    #Remove @ mentions
    df['tweet'] = df['tweet'].progress_apply( lambda t: re.sub(r'@[A-Za-z0-9]+','',t))
    #Remove URLs
    df['tweet'] = df['tweet'].progress_apply(lambda t:re.sub('https?://[A-Za-z0-9./]+','',t))
    #Segment hashtags
    df['tweet'] = df['tweet'].progress_apply(lambda t: segment_hashtags(t))
    #Remove remaining non-alpha characters
    df['tweet'] = df['tweet'].progress_apply(lambda t: re.sub("[^a-zA-Z]", " ", t))
    #Re-label positive tweets with 1 instead of 4
    df['sentiment'] = df['sentiment'].apply(lambda t: 1 if t==4 else t)
    return df


def train_model(df):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['tweet'].values)
    save_tokenizer(tokenizer, 'Chapter05/twitter_tokenizer.pkl')
    X = tokenizer.texts_to_sequences(df['tweet'].values)
    X = pad_sequences(X)
    Y = df['sentiment'].values
    X_train, X_test, Y_train, Y_test = \
    train_test_split(X,Y, test_size=0.20,
    random_state=42,
    stratify=df['sentiment'])
    model = Sequential()
    optimizer = tf.keras.optimizers.Adam(0.00001)
    model.add(Embedding(MAX_NUM_WORDS, EMBEDDING_DIM,
    input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.5,
    recurrent_dropout=0.5,
    return_sequences=True))
    model.add(LSTM(100, dropout=0.5,
    recurrent_dropout=0.5))
    model.add(Dense(1, activation='sigmoid'))
    loss='binary_crossentropy' #Binary in this case
    model.compile(loss=loss, optimizer=optimizer,
    metrics=['accuracy'])
    epochs = 15
    batch_size = 32
    es = [EarlyStopping(monitor='val_loss',
    patience=3, min_delta=0.0001)]
    history = model.fit(X_train, Y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.3,
    callbacks=es)
    accr = model.evaluate(X_test,Y_test)
    print('Test set\n Loss: {:0.3f}\n \
        Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    model.save('Chapter05/twitter_model.h5')
    evaluate(model, X_test, Y_test)
    plot_model(history)


df = get_data(twitter_csv)
df = clean_data(df)
train_model(df)
