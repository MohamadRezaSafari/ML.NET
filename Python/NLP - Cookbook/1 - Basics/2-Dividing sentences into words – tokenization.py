import os
import nltk
import spacy


curerntPath = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(curerntPath, 'sherlock_holmes_1.txt')


file = open(filename, "r", encoding="utf-8")
text = file.read()

# text = text.replace("\n", " ")
# words = nltk.tokenize.word_tokenize(text)

# tweet = "@EmpireStateBldg Central Park Tower is reaaaally hiiiigh"
# tweetWords = nltk.tokenize.casual.casual_tokenize(tweet, 
#                     preserve_case=True,
#                     reduce_len=True,
#                     strip_handles=True)

# print(tweetWords)

text = text.replace("\n", " ")
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
words = [token.text for token in doc]

print(words)
