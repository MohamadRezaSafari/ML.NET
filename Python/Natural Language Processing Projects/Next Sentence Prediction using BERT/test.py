import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
sys.path.append('models')
from official.nlp.data import classifier_data_lib
from official.nlp.bert import tokenization
from official.nlp import optimization
 
# keras imports
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model


curerntPath = os.path.dirname(os.path.abspath(__file__))
# df = pd.read_csv(
#   'https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip',
#                  compression='zip')
df = pd.read_csv(dataset_dir = os.path.join(curerntPath, 'train.csv.zip'), compression='zip')
df.head()

df.target.plot(kind='hist', title='Sincere (0) vs Insincere (1) distribution')