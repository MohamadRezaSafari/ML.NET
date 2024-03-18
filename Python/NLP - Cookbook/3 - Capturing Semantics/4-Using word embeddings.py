import os
from gensim.models import KeyedVectors
import numpy as np


curerntPath = os.path.dirname(os.path.abspath(__file__))
w2vec_model_path = os.path.join(curerntPath, "model.bin")


model = KeyedVectors.load_word2vec_format(w2vec_model_path, binary=True)

# print(model['holmes'])
# print(model.most_similar(['holmes'], topn=15))

# sentence = "It was not that he felt any emotion akin to love for Irene Adler."


# def get_word_vectors(sentence, model):
#     word_vectors = []
#     for word in sentence:
#         try:
#             word_vector = model.get_vector(word.lower())
#             word_vectors.append(word_vector)
#         except KeyError:
#             continue
#     return word_vectors


# def get_sentence_vector(word_vectors):
#     matrix = np.array(word_vectors)
#     centroid = np.mean(matrix[:,:], axis=0)
#     return centroid


# word_vectors = get_word_vectors(sentence, model)
# sentence_vector = get_sentence_vector(word_vectors)
# print(sentence_vector)


words = ['banana', 'apple', 'computer', 'strawberry']

print(model.doesnt_match(words))


word = "cup"
words = ['glass', 'computer', 'pencil', 'watch']
print(model.most_similar_to_given(word, words))
