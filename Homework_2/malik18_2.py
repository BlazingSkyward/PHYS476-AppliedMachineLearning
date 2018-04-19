#Malik Abu-Kalokoh, Mouse

import os
import sys
#Savng all outputs and redirecting to surpress unneeded output
stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow
sys.stderr = stderr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

np.random.seed(7)
#Grabbing filename
filename = sys.argv[1]
names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang', \
         'oldpeak','slope','ca','thal','num']
dataset = pd.read_csv(filename,header=None, names=names)

#Too Much Missing data to matter in ML
del dataset['slope']
del dataset['ca']
del dataset['thal']


dataset = dataset.replace('?',np.NaN)
dataset.dropna(inplace=True)

Y = dataset['num']
del dataset['num']
X = dataset
del dataset

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

GNB = GaussianNB()

GNB.fit(X_train, Y_train)


print (GNB.score(X_test,Y_test)*100)
