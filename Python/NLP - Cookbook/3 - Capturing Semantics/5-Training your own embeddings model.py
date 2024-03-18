import os
import gensim
import pickle
from os import listdir
from os.path import isfile, join
import spacy
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import pickle


curerntPath = os.path.dirname(os.path.abspath(__file__))
word2vec_model_path = filename = os.path.join(curerntPath, "word2vec.model")
books_dir = filename = os.path.join(curerntPath, "1025_1853_bundle_archive")

def get_sentences(filename):
    filename = os.path.join(curerntPath, filename)
    file = open(filename, "r", encoding="utf-8")
    text = file.read()
    text = text.replace("\n", " ")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sentence.text for sentence in doc.sents]
    return sentences


sentences = get_sentences("sherlock_holmes_1.txt")


def get_all_book_sentences(directory):
    text_files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f)) and ".txt" in f]
    all_sentences = []
    for text_file in text_files:
        sentences = get_sentences(text_file)
        all_sentences = all_sentences + sentences
    return all_sentences


def train_word2vec(words, word2vec_model_path):
    model = gensim.models.Word2Vec(words, window=5, size=200)
    model.train(words, total_examples=len(words), epochs=200)
    pickle.dump(model, open(word2vec_model_path, 'wb'))
    return model


sentences = get_all_book_sentences(books_dir)
sentences = [tokenize_nltk(s.lower()) for s in sentences]
model = train_word2vec(sentences, word2vec_model_path)


w1 = "river"
words = model.wv.most_similar(w1, topn=10)
print(words)


model = pickle.load(open(word2vec_model_path, 'rb'))
(analogy_score, word_list) = model.wv.evaluate_word_analogies(datapath('questionswords.txt'))
print(analogy_score)

pretrained_model_path = "Chapter03/40/model.bin"
pretrained_model = \
KeyedVectors.load_word2vec_format(pretrained_model_path,
binary=True)
(analogy_score, word_list) = \
pretrained_model.evaluate_word_
analogies(datapath('questions-words.txt'))
print(analogy_score)