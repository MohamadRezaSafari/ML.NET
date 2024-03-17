import os
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import spacy
from nltk.probability import FreqDist


curerntPath = os.path.dirname(os.path.abspath(__file__))
stemmer = SnowballStemmer('english')
stopwords_file_path = os.path.join(curerntPath, "stopwords.csv")
sherlock_holmes_file_path = os.path.join(curerntPath, "sherlock_holmes_1.txt")

def get_sentences(filename):
    file = open(filename, "r", encoding="utf-8")
    text = file.read()
    text = text.replace("\n", " ")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sentence.text for sentence in doc.sents]
    return sentences

sentences = get_sentences(sherlock_holmes_file_path)


def tokenize_and_stem(sentence):
    tokens = nltk.word_tokenize(sentence)
    filtered_tokens = [t for t in tokens if t not in \
    string.punctuation]
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def read_in_csv(filename):
    file = open(filename, "r", encoding="utf-8")
    text = file.read()
    text = text.replace("\n", " ")
    words = nltk.tokenize.word_tokenize(text)
    freq_dist = FreqDist(word.lower() for word in words)
    words_with_frequencies = [(word, freq_dist[word]) for word in freq_dist.keys()]
    sorted_words = sorted(words_with_frequencies, key=lambda tup: tup[1])
    length_cutoff = int(0.02*len(sorted_words))
    stopwords = [tuple[0] for tuple in sorted_words[-length_cutoff:]]
    return stopwords

stopword_list = read_in_csv(stopwords_file_path)
# stemmed_stopwords = [tokenize_and_stem(stopword)[0] for stopword in stopword_list]
# stopword_list = stopword_list + stemmed_stopwords

# print(stopword_list)

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, max_features=200000,
min_df=0.05, stop_words=stopword_list,
use_idf=True,tokenizer=tokenize_and_stem,
ngram_range=(1,3))
tfidf_vectorizer = tfidf_vectorizer.fit(sentences)

tfidf_matrix = tfidf_vectorizer.transform(sentences)
# print(tfidf_matrix)

dense_matrix = tfidf_matrix.todense()
# print(tfidf_vectorizer.get_feature_names_out())


analyze = tfidf_vectorizer.build_analyzer()
# print(analyze("To Sherlock Holmes she is always _the_woman."))



tfidf_char_vectorizer = TfidfVectorizer(analyzer='char_wb',
    max_df=0.90,
    max_features=200000,
    min_df=0.05,
    use_idf=True,
    ngram_range=(1,3))

tfidf_char_vectorizer = tfidf_char_vectorizer.fit(sentences)
tfidf_matrix = tfidf_char_vectorizer.transform(sentences)
# print(tfidf_matrix)
dense_matrix = tfidf_matrix.todense()
# print(tfidf_char_vectorizer.get_feature_names_out())

analyze = tfidf_char_vectorizer.build_analyzer()
print(analyze("To Sherlock Holmes she is always _the_woman."))
