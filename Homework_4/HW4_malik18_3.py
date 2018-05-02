#Malik Abu-Kalokoh, 

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
from sklearn.preprocessing import normalize
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

learning_rate = 0.001
# Input and target placeholders
inputs_ = tf.placeholder(tf.float32, (None, h,w,d), name="input")
targets_ = tf.placeholder(tf.float32, (None, h,w,d), name="target")

### Encoder
conv1 = tf.layers.conv2d(inputs=inputs_, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now hxwx16
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
# Now h/2xw/2x16
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now h/2xw/2x8
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
# Now h/4xw/4x8
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now h/4xw/4x8
encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
# Now 4x4x8

### Decoder
upsample1 = tf.image.resize_images(encoded, size=(int(h/4),int(w/4)), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 7x7x8
conv4 = tf.layers.conv2d(inputs=upsample1, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 7x7x8
upsample2 = tf.image.resize_images(conv4, size=(int(h/2),int(w/2)), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 14x14x8
conv5 = tf.layers.conv2d(inputs=upsample2, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 14x14x8
upsample3 = tf.image.resize_images(conv5, size=(int(h),int(w)), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 28x28x8
conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 28x28x16

logits = tf.layers.conv2d(inputs=conv6, filters=int(d), kernel_size=(3,3), padding='same', activation=None)
#Now 28x28x1

# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits)

# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)

# Get cost and define the optimizer
cost = tf.metrics.mean_absolute_error(loss,inputs_)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)


learning_rate = 0.0001
epochs = 10
batch_size = 10

init = tf.global_variables_initializer()

X_train, X_test = train_test_split(images,test_size=0.1)
print("about to train")
with tf.Session() as sess:
    # initialise the variables
    sess.run(init)
    total_batch = int(len(X_train) / batch_size)
    for e in range(epochs):
        for ii in range(len(X_train)//batch_size):
            batch = shuffle(X_train,n_samples=batch_size)
            imgs = batch[0].reshape((-1, h, w, d))
            batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: imgs,
                                                         targets_: imgs})

            print("Epoch: {}/{}...".format(e+1, epochs),"Training loss: {:.4f}".format(batch_cost))


"python3 HW4_malik18_3.py ../../justinkterry/data/flowers"