import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
stdout = sys.stdout
sys.stdout = open('/dev/null', 'w')

from keras.models import Sequential
from keras.layers import Dense
import tensorflow
import pandas
import numpy
numpy.random.seed(7)

data = pandas.read_csv('abalone.csv')
data = pandas.get_dummies(data=data, columns=['Sex'])
test = data[:400]
train = data[400:]

y = train['Rings']
del train['Rings']
x = train.values

y_test = test['Rings']
del test['Rings']
x_test = test.values

model = Sequential()
model.add(Dense(12,input_dim=10,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='linear'))

model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['MAE'])
model.fit(x,y,epochs=25,batch_size=10)
rings=model.evaluate(x_test,y_test)
print "\n%s: %.2f" % (model.metrics_names[1], rings[1])
