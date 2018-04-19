#Malik Abu-Kalokoh

import os
import sys
#Savng all outputs and redirecting to surpress unneeded output
stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow
#from sklearn.model_selection import train_test_split
import random
from math import floor
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv1D, Flatten
import pandas
import numpy
numpy.random.seed(7)
#Redirect to a proper output
sys.stderr = stderr

filename = sys.argv[1] #Should be file for beadekris.data

names = ['Sepal Length','Sepal Width','Petal Length','Petal Width','Class']
data = pandas.read_csv(filename, names=names,keep_default_na=False)

#Shuffle Data
numpy.random.shuffle(data.values)
split = 55

output = data['Class']
#One Hot Encoding
output = pandas.get_dummies(output, columns=['Class'])
del data['Class']

y = output[split:]
x = data[split:]

y_test = output[:split]
x_test = data[:split]

model = Sequential()
model.add(Dense(20,input_dim=4,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(3,activation='relu'))

model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=50,batch_size=10,verbose=0)
score=model.evaluate(x_test,y_test,verbose=0)

print(score[1])
