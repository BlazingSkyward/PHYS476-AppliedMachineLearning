from sklearn import svm
import pandas as pd
import random
from math import floor
import numpy as np
np.random.seed(42)

if __name__ == "__main__":
    filename = 'pima-indians-diabetes.data'
    names = ['Number of times pregnant', 'Plasma glucose concentration a 2 hours in an oral glucose tolerance test', 'Diastolic blood pressure', 'Triceps skin fold thickness', '2-Hour serum insulin', 'Body mass index', 'Diabetes pedigree function', 'Age', 'Class']
    data = pd.read_csv(filename,names=names)

    np.random.shuffle(data.values)
    split = random.randrange(floor(.5*len(data.index)),floor(.8*len(data.index)))

    #Splitting Data
    test = data[:split]
    train = data[split:]

    y = train['Class'].values
    del train['Class']
    x = train.values

    y_test = test['Class'].values
    del test['Class']
    x_test = test.values

    clf = svm.SVC(kernel='linear',gamma=0.75,C=15)
    clf.fit(x,y)

    prediction = clf.predict(test)
    correct = 0

    for i in range(0,len(prediction)):
        if prediction[i] == y_test[i]:
            correct += 1

    print((correct*100)/len(y_test))
