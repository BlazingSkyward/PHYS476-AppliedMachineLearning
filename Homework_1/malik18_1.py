#Malik Abu-Kalokoh, Breast Cancer Wisonconsin

import os
import sys
#Savng all outputs and redirecting to surpress unneeded output
stderr = sys.stderr
stdout = sys.stdout
sys.stdout = open('/dev/null', 'w')
sys.stderr = open('/dev/null', 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow
#from sklearn.model_selection import train_test_split
import random
from math import floor
from keras.models import Sequential
from keras.layers import Dense
import pandas
import numpy
numpy.random.seed(7)

#Redirect to a proper output
sys.stderr = stderr
sys.stdout = stdout
filename = sys.argv[1] #Should be file for breast cancer

#names
names = ['Sample code number','Clump Thickness', 'Uniformity of Cell Size','Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
data = pandas.read_csv(filename, names=names,keep_default_na=False)

#Shuffle Data
numpy.random.shuffle(data.values)
split = 354

#Remove messed up data
del data['Sample code number']
data = data.replace('?',0)

output = data['Class']
#One Hot Encoding
output = pandas.get_dummies(output, columns=['Class'])
del data['Class']

y = output[split:]
x = data[split:]

y_test = output[:split]
x_test = data[:split]

model = Sequential()
model.add(Dense(20,input_dim=9,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(2,activation='relu'))

model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=50,batch_size=10,verbose=0)
score=model.evaluate(x_test,y_test,verbose=0)

print(score[1])
