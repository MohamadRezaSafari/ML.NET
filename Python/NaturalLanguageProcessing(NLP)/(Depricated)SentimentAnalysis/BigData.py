import os
import numpy as np
import re
from nltk.corpus import stopwords

stop = stopwords.words('english')
curerntPath = os.path.dirname(os.path.abspath(__file__))

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
    text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) \
    + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

next(stream_docs(path=os.path.join(curerntPath, 'movie_data.csv')))