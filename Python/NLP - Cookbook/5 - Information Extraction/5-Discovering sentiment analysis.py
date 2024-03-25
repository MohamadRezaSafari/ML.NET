from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


nltk.download('vader_lexicon')
sentences = ["I love going to school!", "I hate going to school!"]
sid = SentimentIntensityAnalyzer()


def get_blob_sentiment(sentence):
    result = TextBlob(sentence).sentiment
    print(sentence, result.polarity)
    return result.polarity


def get_nltk_sentiment(sentence):
    ss = sid.polarity_scores(sentence)
    print(sentence, ss['compound'])
    return ss['compound']


for sentence in sentences:
    sentiment = get_nltk_sentiment(sentence)
