#Malik Abu-Kalokoh, 

import os
import sys

#Savng all outputs and redirecting to surpress unneeded output
stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle

sys.stderr = stderr
np.random.seed(42)

filename = sys.argv[1] #should be adult.data
names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation',
            'relationship', 'race','sex','capital-gain','capital-loss','hours-per-week','native-country']
