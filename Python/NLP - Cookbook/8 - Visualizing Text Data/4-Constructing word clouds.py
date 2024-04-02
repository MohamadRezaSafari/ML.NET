import os
import nltk
from os import path
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.probability import FreqDist
import numpy as np
from PIL import Image
# from Chapter01.dividing_into_sentences import read_text_file
# from Chapter01.removing_stopwords import compile_stopwords_list_frequency


curerntPath = os.path.dirname(os.path.abspath(__file__))


def read_text_file(text_file):
    file = open(text_file, "r", encoding="utf-8")
    text = file.read()
    text = text.replace("\n", " ")
    # nlp = spacy.load("en_core_web_sm")
    # doc = nlp(text)
    # sentences = [sentence.text for sentence in doc.sents]
    return text

def compile_stopwords_list_frequency(text):
    words = nltk.tokenize.word_tokenize(text)
    freq_dist = FreqDist(word.lower() for word in words)
    words_with_frequencies = [(word, freq_dist[word]) for word in freq_dist.keys()]
    sorted_words = sorted(words_with_frequencies, key=lambda tup: tup[1])
    length_cutoff = int(0.02*len(sorted_words))
    # stopwords = [tuple[0] for tuple in sorted_words if tuple[1] > 100]
    stopwords = [tuple[0] for tuple in sorted_words[-length_cutoff:]]

def create_wordcloud(text, stopwords, filename):
    wordcloud = WordCloud(min_font_size=10, max_font_size=100,
        stopwords=stopwords, width=1000,
        height=1000, max_words=1000,
        background_color="white").generate(text)
    wordcloud.to_file(filename)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


text_file = os.path.join(curerntPath, 'sherlock_holmes_1.txt')
text = read_text_file(text_file)


create_wordcloud(text, compile_stopwords_list_frequency(text), os.path.join(curerntPath, 'sherlock_wc.png'))


def create_wordcloud(text, stopwords, filename, apply_mask=None):
    if (apply_mask is not None):
        wordcloud = WordCloud(background_color="white",
            max_words=2000,
            mask=apply_mask,
            stopwords=stopwords,
            min_font_size=10,
            max_font_size=100)
        wordcloud.generate(text)
        wordcloud.to_file(filename)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.figure()
        plt.imshow(apply_mask, cmap=plt.cm.gray,
        interpolation='bilinear')
        plt.axis("off")
        plt.show()
    else:
        wordcloud = WordCloud(min_font_size=10,
            max_font_size=100,
            stopwords=stopwords,
            width=1000,
            height=1000,
            max_words=1000,
            background_color="white").generate(text)
        wordcloud.to_file(filename)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()


sherlock_data = Image.open(os.path.join(curerntPath, 'sherlock_wc.png'))
sherlock_mask = np.array(sherlock_data)
create_wordcloud(text, compile_stopwords_list_frequency(text), os.path.join(curerntPath, 'sherlock_mask.png'), apply_mask=sherlock_mask)

