import tarfile
import os
import pyprind
import pandas as pd
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

curerntPath = os.path.dirname(os.path.abspath(__file__))
# gzFile = os.path.join(curerntPath, "aclImdb_v1.tar.gz")
basepath = os.path.join(curerntPath, 'aclImdb')

# with tarfile.open(gzFile, 'r:gz') as tar:
#     tar.extractall()

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000, stream=sys.stdout)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            # df = df.append([[txt, labels[l]]], ignore_index=True)
            df = pd.Series([[txt, labels[l]]])
            pbar.update()
df.columns = ['review', 'sentiment']


np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv(os.path.join(curerntPath, 'movie_data.csv'), index=False, encoding='utf-8')

df = pd.read_csv(os.path.join(curerntPath, 'movie_data.csv'), encoding='utf-8')
df = df.rename(columns={"0": "review", "1": "sentiment"})
df.head(3)
df.shape

count = CountVectorizer()
docs = np.array(['The sun is shining', 'The weather is sweet',
                 'The sun is shining, the weather is sweet,'
                 'and one and one is two'])
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())


tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

df.loc[0, 'review'][-50:]

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
    text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
    ' '.join(emoticons).replace('-', ''))
    return text

preprocessor(df.loc[0, 'review'][-50:])
preprocessor("</a>This :) is :( a test :-)!")
df['review'] = df['review'].apply(preprocessor)

def tokenizer(text):
    return text.split()

tokenizer('runners like running and thus they run')

porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

tokenizer_porter('runners like running and thus they run')
nltk.download('stopwords')

stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot') if w not in stop]


X_train = df.loc[:25000, 'review'].values
y_train = df.iloc[:25000, 0].values
X_test = df.loc[25000:, 'review'].values
y_test = df.iloc[25000:, 0].values


tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

small_param_grid = [
    {
    'vect__ngram_range': [(1, 1)],
    'vect__stop_words': [None],
    'vect__tokenizer': [tokenizer, tokenizer_porter],
    'clf__penalty': ['l2'],
    'clf__C': [1.0, 10.0]
    },
    {
    'vect__ngram_range': [(1, 1)],
    'vect__stop_words': [stop, None],
    'vect__tokenizer': [tokenizer],
    'vect__use_idf':[False],
    'vect__norm':[None],
    'clf__penalty': ['l2'],
    'clf__C': [1.0, 10.0]
    },
]

lr_tfidf = Pipeline([
    ('vect', tfidf),
    ('clf', LogisticRegression(solver='liblinear'))
])

gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid,
scoring='accuracy', cv=5,
verbose=2, n_jobs=1)
gs_lr_tfidf.fit(X_train, y_train)

print(f'Best parameter set: {gs_lr_tfidf.best_params_}')
print(f'CV Accuracy: {gs_lr_tfidf.best_score_:.3f}')
clf = gs_lr_tfidf.best_estimator
print(f'Test Accuracy: {clf.score(X_test, y_test):.3f}')
