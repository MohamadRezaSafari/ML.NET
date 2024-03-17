import os
from sklearn.feature_extraction.text import CountVectorizer
import spacy


curerntPath = os.path.dirname(os.path.abspath(__file__))


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
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
X = bigram_vectorizer.fit_transform(sentences)
# print(X)

denseX = X.todense()
# print(denseX)

# print(bigram_vectorizer.get_feature_names_out())


new_sentence = "I had seen little of Holmes lately."
new_sentence_vector = bigram_vectorizer.transform([new_sentence])

# print(new_sentence_vector)
# print(new_sentence_vector.todense())

# vectorizer = CountVectorizer(stop_words='english')
# new_sentence1 = " And yet there was but one woman to him, and that woman was the late Irene Adler, of dubious and questionable memory."
# new_sentence_vector1 = vectorizer.transform([new_sentence1])

# print(new_sentence_vector1)
# print(new_sentence_vector1.todense())
