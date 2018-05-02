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


h, w, d = images[0].shape
images, labels = shuffle(images, labels)

X_train, X_test, Y_train, Y_test = train_test_split(images,labels,test_size=0.1)
image_flatsize = h*w*d

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, 
                               padding='SAME')

    return out_layer


# Parameters
num_steps = 2000
learning_rate = 0.0001
epochs = 10
batch_size = 10

X = tf.placeholder(tf.float32, shape=[None,h*w*d])
X_image = tf.reshape(X,shape=[-1,h,w,d]) 

num_classes = len(flower_types)

Y = tf.placeholder(tf.float32, [None,num_classes])

conv1_1 = create_new_conv_layer(X_image, 3, 32, [5, 5], [2, 2], name='conv1_1')
conv1_2 = create_new_conv_layer(conv1_1, 32, 64, [5, 5], [2, 2], name='conv1_2')

flattened = tf.reshape(conv1_2, [-1, int(h/4) * int(w/4) * 64])
wd1 = tf.Variable(tf.truncated_normal([int(h/4) * int(w/4) * 64, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

wd2 = tf.Variable(tf.truncated_normal([1000, num_classes], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([num_classes], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=Y))

optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # initialise the variables
    sess.run(init)
    total_batch = int(len(Y_train) / batch_size)
    for epoch in range(epochs):
        X_train, Y_train = shuffle(X_train,Y_train)
        avg_cost = 0
        for i in range(0,len(Y_train),batch_size):
          if (batch_size + i) < len(Y_train): 
            end = i + batch_size
          else:
            end = len(Y_train) - 1
          
          batch_x = X_train[i:end].reshape(end - i,image_flatsize) 
          batch_y = Y_train[i:end]
          _, c = sess.run([optimiser, cross_entropy], feed_dict={X: batch_x, Y: batch_y})
          avg_cost += c / total_batch
        test_acc = sess.run(accuracy, feed_dict={X: X_test.reshape(len(Y_test),image_flatsize), Y: Y_test})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "test accuracy: {:.3f}".format(test_acc))

    print("\nTraining complete!")
    print(sess.run(accuracy, feed_dict={X: X_test.reshape(len(Y_test),image_flatsize), Y: Y_test}))

