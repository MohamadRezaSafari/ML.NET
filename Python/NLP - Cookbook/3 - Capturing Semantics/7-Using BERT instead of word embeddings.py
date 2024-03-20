import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
import spacy
# from Chapter01.dividing_into_sentences import read_text_ file, divide_into_sentences_nltk


curerntPath = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(curerntPath, 'sherlock_holmes_1.txt')
file = open(filename, "r", encoding="utf-8")
text = file.read()

text = text.replace("\n", " ")
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
sentences = [sentence.text for sentence in doc.sents]

model = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = model.encode(sentences)

sentence_embeddings = model.encode(["the beautifullake"])
print(sentence_embeddings)
