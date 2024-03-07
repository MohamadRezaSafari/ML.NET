from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()

words = ['duck', 'geese', 'cats', 'books']
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

print(lemmatized_words)
