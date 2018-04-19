"""
Malik Abu-Kalokoh
"""
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import tensorflow as tf
sys.stderr = stderr

import pandas as pd
import numpy as np

# fix random seed
seed = 7
np.random.seed(seed)

#import data
image  = pd.read_csv(sys.argv[1], header = None)
labels = pd.read_csv(sys.argv[2], header = None)

#build training and testing sets
train_in = image.sample(frac = 9/10, axis = 0).sort_index()
test_in  = image.drop(train_in.index).sort_index()
del image

train_out = labels.drop(test_in.index).sort_index()
test_out  = labels.drop(train_in.index).sort_index()
del labels

#munge data
train_in  = train_in.as_matrix().reshape(len(train_in.index), 1, 28, 28).astype('float32')
test_in   = test_in.as_matrix().reshape(len(test_in.index), 1, 28, 28).astype('float32')
train_out = np_utils.to_categorical(train_out)
test_out  = np_utils.to_categorical(test_out)

#build model
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(1, 28, 28), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train model
model.fit(train_in, train_out, epochs = 1, batch_size = 200,verbose=0)

#test model
scores = model.evaluate(test_in, test_out, verbose = 0)

#output results
print(scores[1])
