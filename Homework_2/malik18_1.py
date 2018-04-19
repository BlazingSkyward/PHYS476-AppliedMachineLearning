#Malik Abu-Kalokoh, Mouse

import os
import sys
#Savng all outputs and redirecting to surpress unneeded output
stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#Redirect to a proper output
sys.stderr = stderr

np.random.seed(7)
#Grabbing filename
filename = sys.argv[1]

dataset  = pd.read_excel(filename)
del dataset['MouseID']

#Replace missing values with mean of each column
dataset.fillna(dataset.mean(), inplace=True)

Y = dataset['class']
del dataset['class']
X = dataset
del dataset

#One-Hotencoding
X = pd.get_dummies(X,columns=['Genotype','Treatment','Behavior'])
Y = pd.get_dummies(Y,columns=['class'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

KNN = KNeighborsClassifier()

KNN.fit(X_train, Y_train)

print (KNN.score(X_test,Y_test)*100)
