import os
import nltk
import spacy


curerntPath = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(curerntPath, 'sherlock_holmes_1.txt')
# file = open(filename, "r", encoding="utf-8")

# text = file.read()
# text = text.replace("\n", " ")

# nltk.download('punkt')
# tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
# sentences = tokenizer.tokenize(text)

# print(sentences)



file = open(filename, "r", encoding="utf-8")
text = file.read()

text = text.replace("\n", " ")
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
sentences = [sentence.text for sentence in doc.sents]

print(sentences)
