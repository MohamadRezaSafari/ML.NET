import csv
import os
import nltk
from nltk.probability import FreqDist


curerntPath = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(curerntPath, 'sherlock_holmes_1.txt')


# csv_file = os.path.join(curerntPath, 'stopwords.csv')
# with open(csv_file, 'r', encoding='utf-8') as fp:
#     reader = csv.reader(fp, delimiter=',', quotechar='"')
#     stopwords = [row[0] for row in reader]
#     # stopwords = nltk.corpus.stopwords.words('english')


# file = open(filename, "r", encoding="utf-8")
# text = file.read()

# text = text.replace("\n", " ")
# words = nltk.tokenize.word_tokenize(text)
# words = [word for word in words if word.lower() not in stopwords]

# print(words)


file = open(filename, "r", encoding="utf-8")
text = file.read()

text = text.replace("\n", " ")
words = nltk.tokenize.word_tokenize(text)

freq_dist = FreqDist(word.lower() for word in words)
words_with_frequencies = [(word, freq_dist[word]) for word in freq_dist.keys()]
sorted_words = sorted(words_with_frequencies, key=lambda tup: tup[1])
length_cutoff = int(0.02*len(sorted_words))
# stopwords = [tuple[0] for tuple in sorted_words if tuple[1] > 100]
stopwords = [tuple[0] for tuple in sorted_words[-length_cutoff:]]

print(stopwords)
