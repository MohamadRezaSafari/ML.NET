from nltk.stem.snowball import SnowballStemmer


stemmer = SnowballStemmer('english')
words = ['leaf', 'leaves', 'booking', 'writing',
        'completed', 'stemming', 'skies']
stemmed_words = [stemmer.stem(word) for word in words]

print(stemmed_words)
print(SnowballStemmer.languages)