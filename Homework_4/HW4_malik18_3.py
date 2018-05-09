#Malik Abu-Kalokoh, Flower Autoencoder using CNN and MAE
#https://www.kaggle.com/alxmamaev/flowers-recognition

import os
import sys
import glob
#Savng all outputs and redirecting to surpress unneeded output
stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.contrib.layers as lays
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle

sys.stderr = stderr
np.random.seed(42)
TESTING = False

filename = sys.argv[1] #Should be parent directory of flowers 
flowers_path = filename
flower_types = os.listdir(flowers_path)
if '.DS_Store' in flower_types: flower_types.remove('.DS_Store')

def resize_with_pad(img, img_size): 
    height, width, _ = img.shape
    ratio = img_size / max(height, width)
    if ratio < 1:
        img = cv2.resize(img, (int(ratio * width), int(ratio * height)))
    padding = ((img_size - img.shape[0], 0), (img_size - img.shape[1],0), (0,0))
    
    return np.pad(img, padding, 'constant')


images = []

for species in flower_types:
    index = flower_types.index(species)
    # Get all the file names
    all_flowers_path = os.path.join(flowers_path,species ,'*g')
    all_flowers = glob.glob(all_flowers_path)
    # Add them to the list
    for flower in all_flowers:
        img = cv2.imread(flower)
        img = resize_with_pad(img,112)
        images.append(img)

images = np.array(images)


np.random.shuffle(images)
h, w, d = images[0].shape
n_features = h*w*d

mean, std = np.mean(images), np.std(images)

def preprocess(img):
    norm_img = (img - mean) / std
    return norm_img 

images = preprocess(images)

learning_rate = 0.01
# Input and target placeholders
inputs_ = tf.placeholder(tf.float32, (None, h,w,d), name="input")
targets_ = tf.placeholder(tf.float32, (None, h,w,d), name="target")

net = lays.conv2d(inputs_, 32, [5, 5], stride=2, padding='SAME')
net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
net = lays.conv2d(net, 8, [5, 5], stride=4, padding='SAME')

net = lays.conv2d_transpose(net, 16, [5, 5], stride=4, padding='SAME')
net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
net = lays.conv2d_transpose(net, 3, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)

loss = tf.reduce_mean(tf.abs(net - inputs_))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

learning_rate = 0.01
epochs = 30
batch_size = 50

init = tf.global_variables_initializer()

X_train, X_test = train_test_split(images,test_size=0.1)

def batching(data,batch_size):
    temp = shuffle(data)
    for i in range(0,len(temp),batch_size):
        if (batch_size + i) < len(temp): 
            end = i + batch_size
        else:
            end = len(temp) - 1
        yield temp[i:end]
        

if TESTING:
    print("about to train")
with tf.Session() as sess:
    # initialise the variables
    sess.run(init)
    total_batch = int(len(X_train) / batch_size)
    for e in range(epochs):
        avg_cost = 0
        for curr in batching(X_train,batch_size):
            curr = curr.reshape(-1,h,w,d)
            _, batch_cost = sess.run([train_op, loss], feed_dict={inputs_: curr})
            avg_cost += batch_cost/total_batch
        if TESTING:
            print("Epoch: {}/{}...".format(e+1, epochs),"Training loss: {:.4f}".format(avg_cost))

    if TESTING:
        print("Testing Accuracy:", 1 - sess.run(loss, feed_dict={inputs_: X_test.reshape((-1, h, w, d))}))
    else: 
        print(1 - sess.run(loss, feed_dict={inputs_: X_test.reshape((-1, h, w, d))}))