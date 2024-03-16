import os
import spacy
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

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


def create_vectorizer(sentences):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    return (vectorizer, X)


sentences = get_sentences("sherlock_holmes_1.txt")
(vectorizer, X) = create_vectorizer(sentences)
denseX = X.todense()

# print(vectorizer.get_feature_names_out())


new_sentence = "I had seen little of Holmes lately."
new_sentence_vector = vectorizer.transform([new_sentence])

# print(new_sentence_vector)


filename = os.path.join(curerntPath, 'sherlock_holmes_1.txt')
file = open(filename, "r", encoding="utf-8")
text = file.read()
text = text.replace("\n", " ")

nltk.download('punkt')
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
sentences = tokenizer.tokenize(text)

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(sentences)
# print(vectorizer.get_feature_names_out())


new_sentence = "And yet there was but one woman to him, and that woman was the late Irene Adler, of dubious and questionable memory."
new_sentence_vector = vectorizer.transform([new_sentence])
analyze = vectorizer.build_analyzer()
print(analyze(new_sentence))
