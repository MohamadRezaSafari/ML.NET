# linear algebra 
import numpy as np 
# data processing, CSV file I / O (e.g. pd.read_csv) 
import pandas as pd 
import os 
import tensorflow as tf 
from keras.preprocessing.sequence import pad_sequences 
from keras.preprocessing.text import Tokenizer 
from keras.models import Model 
from keras.layers import Flatten, Dense, LSTM, Dropout, Embedding, Activation 
from keras.layers import concatenate, BatchNormalization, Input
from keras.layers.merge import add 
from keras.utils import to_categorical, plot_model 
from keras.applications.inception_v3 import InceptionV3, preprocess_input 
import matplotlib.pyplot as plt # for plotting data 
import cv2 



def load_description(text): 
	mapping = dict() 
	for line in text.split("\n"): 
		token = line.split("\t") 
		if len(line) < 2: # remove short descriptions 
			continue
		img_id = token[0].split('.')[0] # name of the image 
		img_des = token[1]			 # description of the image 
		if img_id not in mapping: 
			mapping[img_id] = list() 
		mapping[img_id].append(img_des) 
	return mapping 

token_path = '/kaggle / input / flickr8k / flickr_data / Flickr_Data / Flickr_TextData / Flickr8k.token.txt'
text = open(token_path, 'r', encoding = 'utf-8').read() 
descriptions = load_description(text) 
print(descriptions['1000268201_693b08cb0e'])
