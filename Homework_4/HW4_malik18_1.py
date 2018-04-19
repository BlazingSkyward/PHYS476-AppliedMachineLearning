#Malik Abu-Kalokoh,

import os
import sys
#Savng all outputs and redirecting to surpress unneeded output
stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.contrib import rnn
from random import shuffle
sys.stderr = stderr
np.random.seed(42)


filename = sys.argv[1] #Should be file for breast cancer

dataset = pd.read_csv(filename, keep_default_na=True, na_values=' ')
del dataset['Date'] #Date can be used to interpret other things
del dataset['OpenInt'] #Stock is not open international

output = dataset['Close']
del dataset['Close']

#Split them in a correct length

input_size = 1
num_steps = 30
lstm_size = 128
num_layers = 1
keep_prob = 0.8

batch_size = 30
init_learning_rate = 0.001
learning_rate_decay = 0.99
init_epoch = 5
max_epoch = 50

