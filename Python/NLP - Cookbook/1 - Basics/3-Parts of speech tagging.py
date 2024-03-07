import os
import nltk
import spacy


curerntPath = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(curerntPath, 'sherlock_holmes_1.txt')


file = open(filename, "r", encoding="utf-8")
text = file.read()

# text = text.replace("\n", " ")
# nlp = spacy.load("en_core_web_sm")
# doc = nlp(text)

# words = [token.text for token in doc]
# pos = [token.pos_ for token in doc]
# word_pos_tuples = list(zip(words, pos))

# print(word_pos_tuples)


text = text.replace("\n", " ")
words = nltk.tokenize.word_tokenize(text)
words_with_pos = nltk.pos_tag(words)
