#Malik Abu-Kalokoh, CNN on Flower Data

import os
import sys
import glob
#Savng all outputs and redirecting to surpress unneeded output
stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

sys.stderr = stderr
np.random.seed(42)

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
labels = [] 

for species in flower_types:
    index = flower_types.index(species)
    # Get all the file names
    all_flowers_path = os.path.join(flowers_path,species ,'*g')
    all_flowers = glob.glob(all_flowers_path)
    # Add them to the list
    for flower in all_flowers:
        img = cv2.imread(flower)
        img = resize_with_pad(img,160)
        images.append(img)

        label = np.zeros(len(flower_types))
        label[index] = 1.0
        labels.append(label)

images = np.array(images)
labels = np.array(labels)



images, labels = shuffle(images, labels)
h, w, d = images[0].shape
image_flatsize = h*w*d
num_classes = len(flower_types)

X_train, X_test, Y_train, Y_test = train_test_split(images,labels,test_size=0.1)

BATCH_SIZE = 10
USE_RELU = True

def weight_variable(shape):
    # From the mnist tutorial
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def fc_layer(previous, input_size, output_size):
    W = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    return tf.matmul(previous, W) + b


def autoencoder(x):
    # first fully connected layer with 50 neurons using tanh activation
    l1 = tf.nn.tanh(fc_layer(x, image_flatsize, 50))
    # second fully connected layer with 50 neurons using tanh activation
    l2 = tf.nn.tanh(fc_layer(l1, 50, 50))
    # third fully connected layer with 2 neurons
    l3 = fc_layer(l2, 50, 2)
    # fourth fully connected layer with 50 neurons and tanh activation
    l4 = tf.nn.tanh(fc_layer(l3, 2, 50))
    # fifth fully connected layer with 50 neurons and tanh activation
    l5 = tf.nn.tanh(fc_layer(l4, 50, 50))
    # readout layer
    if USE_RELU:
        out = tf.nn.relu(fc_layer(l5, 50, image_flatsize))
    else:
        out = fc_layer(l5, 50, image_flatsize)
    # let's use an l2 loss on the output image
    loss = tf.reduce_mean(tf.squared_difference(x, out))
    return loss, out, l3


x = tf.placeholder(tf.float32, shape=[None, image_flatsize])
loss, output, latent = autoencoder(x)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(len(Y_train) / BATCH_SIZE)
    for epoch in range(epochs):
        X_train, Y_train = shuffle(X_train,Y_train)
        avg_cost = 0
        for i in range(0,len(Y_train),BATCH_SIZE):
          if (BATCH_SIZE + i) < len(Y_train): 
            end = i + BATCH_SIZE
          else:
            end = len(Y_train) - 1
          
          batch_x = X_train[i:end].reshape(end - i,image_flatsize) 
          batch_y = Y_train[i:end]
          _, c = sess.run([train_step, cross_entropy], feed_dict={X: batch_x, Y: batch_y})
          avg_cost += c / total_batch
        test_acc = sess.run(accuracy, feed_dict={X: X_test.reshape(len(Y_test),image_flatsize), Y: Y_test})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "test accuracy: {:.3f}".format(test_acc))

    print("\nTraining complete!")
    print(sess.run(accuracy, feed_dict={X: X_test.reshape(len(Y_test),image_flatsize), Y: Y_test}))