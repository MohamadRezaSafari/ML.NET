import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split

# Text Pre-processing libraries
import nltk
import string
import warnings
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Tensorflow imports to build the model.
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.tree import DecisionTreeClassifier
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))

curerntPath = os.path.dirname(os.path.abspath(__file__))

nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
warnings.filterwarnings('ignore')


df = pd.read_csv(os.path.join(curerntPath, 'labeled_data.csv'))
# df.head()
# df.shape
# df.info()


# plt.pie(df['class'].value_counts().values,
# 		labels = df['class'].value_counts().index,
# 		autopct='%1.1f%%')
# plt.show()


# Lower case all the words of the tweet before any preprocessing
# df['tweet'] = df['tweet'].str.lower()

# # Removing punctuations present in the text
# punctuations_list = string.punctuation
# def remove_punctuations(text):
# 	temp = str.maketrans('', '', punctuations_list)
# 	return text.translate(temp)

# df['tweet']= df['tweet'].apply(lambda x: remove_punctuations(x))
# df.head()



def remove_stopwords(text):
	stop_words = stopwords.words('english')

	imp_words = []

	# Storing the important words
	for word in str(text).split():

		if word not in stop_words:

			# Let's Lemmatize the word as well
			# before appending to the imp_words list.

			lemmatizer = WordNetLemmatizer()
			lemmatizer.lemmatize(word)

			imp_words.append(word)

	output = " ".join(imp_words)

	return output


# df['tweet'] = df['tweet'].apply(lambda text: remove_stopwords(text))
# df.head()



def plot_word_cloud(data, typ):
	# Joining all the tweets to get the corpus
	email_corpus = " ".join(data['tweet'])

	plt.figure(figsize = (10,10))

	# Forming the word cloud
	wc = WordCloud(max_words = 100,
					width = 200,
					height = 100,
					collocations = False).generate(email_corpus)

	# Plotting the wordcloud obtained above
	plt.title(f'WordCloud for {typ} emails.', fontsize = 15)
	plt.axis('off')
	plt.imshow(wc)
	plt.show()
	print()

# plot_word_cloud(df[df['class']==2], typ='Neither')


# class_2 = df[df['class'] == 2]
# class_1 = df[df['class'] == 1].sample(n=3500)
# class_0 = df[df['class'] == 0]

# balanced_df = pd.concat([class_0, class_0, class_0, class_1, class_2], axis=0)


# plt.pie(balanced_df['class'].value_counts().values,
# 		labels=balanced_df['class'].value_counts().index,
# 		autopct='%1.1f%%')
# plt.show()


df["labels"] = df["class"].map({0: "Hate Speech", 
                                    1: "Offensive Language", 
                                    2: "No Hate and Offensive"})
data = df[["tweet", "labels"]]

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

df['tweet'] = df['tweet'].apply(clean)


x = np.array(df["tweet"])
y = np.array(df["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# features = balanced_df['tweet']
# target = balanced_df['class']

# X_train, X_val, Y_train, Y_val = train_test_split(features,
# 												target,
# 												test_size=0.2,
# 												random_state=22)
# X_train.shape, X_val.shape


# Y_train = pd.get_dummies(Y_train)
# Y_val = pd.get_dummies(Y_val)
# Y_train.shape, Y_val.shape


# max_words = 5000
# max_len = 100

# token = Tokenizer(num_words=max_words,
# 				lower=True,
# 				split=' ')

# token.fit_on_texts(X_train)


# # training the tokenizer
# max_words = 5000
# token = Tokenizer(num_words=max_words,
# 				lower=True,
# 				split=' ')
# token.fit_on_texts(X_train)

# #Generating token embeddings
# Training_seq = token.texts_to_sequences(X_train)
# Training_pad = pad_sequences(Training_seq,
# 							maxlen=50,
# 							padding='post',
# 							truncating='post')

# Testing_seq = token.texts_to_sequences(X_train)
# Testing_pad = pad_sequences(Testing_seq,
# 							maxlen=50,
# 							padding='post',
# 							truncating='post')



# model = keras.models.Sequential([
# 	layers.Embedding(max_words, 32, input_length=max_len),
# 	layers.Bidirectional(layers.LSTM(16)),
# 	layers.Dense(512, activation='relu', kernel_regularizer='l1'),
# 	layers.BatchNormalization(),
# 	layers.Dropout(0.3),
# 	layers.Dense(3, activation='softmax')
# ])

# model.compile(loss='categorical_crossentropy',
# 			optimizer='adam',
# 			metrics=['accuracy'])

# model.summary()



# keras.utils.plot_model(
# 	model,
# 	show_shapes=True,
# 	show_dtype=True,
# 	show_layer_activations=True
# )


# model = keras.models.Sequential([
# 	layers.Embedding(max_words, 32, input_length=max_len),
# 	layers.Bidirectional(layers.LSTM(16)),
# 	layers.Dense(512, activation='relu', kernel_regularizer='l1'),
# 	layers.BatchNormalization(),
# 	layers.Dropout(0.3),
# 	layers.Dense(3, activation='softmax')
# ])

# model.compile(loss='categorical_crossentropy',
# 			optimizer='adam',
# 			metrics=['accuracy'])

# model.summary()


# keras.utils.plot_model(
# 	model,
# 	show_shapes=True,
# 	show_dtype=True,
# 	show_layer_activations=True
# )




# es = EarlyStopping(patience=3,
# 				monitor = 'val_accuracy',
# 				restore_best_weights = True)

# lr = ReduceLROnPlateau(patience = 2,
# 					monitor = 'val_loss',
# 					factor = 0.5,
# 					verbose = 0)




model = DecisionTreeClassifier()

history = model.fit(X_train, y_train,
					# validation_data=(X_val, Y_val),
					# epochs=50,
					# verbose=1,
					# batch_size=32,
					# callbacks=[lr, es]
					)


sample = "Let's unite and kill all the people who are protesting against the government"
data = cv.transform([sample]).toarray()
print(model.predict(data))

# history_df = pd.DataFrame(history.history)
# history_df.loc[:, ['loss', 'val_loss']].plot()
# history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
# plt.show()
