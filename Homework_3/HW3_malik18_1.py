#Malik Abu-Kalokoh,

import os
import sys
#Savng all outputs and redirecting to surpress unneeded output
stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
sys.stderr = stderr
np.random.seed(42)

filename = sys.argv[1] #Should be file for breast cancer

names = ['Sample code number','Clump Thickness', 'Uniformity of Cell Size',
         'Uniformity of Cell Shape', 'Marginal Adhesion',
         'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
         'Normal Nucleoli', 'Mitoses', 'Class']

dataset = pd.read_csv(filename, names=names,keep_default_na=True, na_values='?')

#Remove messed up data
del dataset['Sample code number']
dataset.dropna(inplace=True)

output = dataset['Class']
#One Hot Encoding
output = pd.get_dummies(output, columns=['Class'])
del dataset['Class']
X = dataset
del dataset
Y = output

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# Parameters
learning_rate = 0.1
num_steps = 2000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 15
n_hidden_2 = 14
n_hidden_3 = 13
n_hidden_4 = 12
num_input = len(list(X))
num_classes = len(list(Y))

init = tf.global_variables_initializer()

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_4, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
def neural_net(x):
    out1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    out1 = tf.nn.relu(out1)
    out2 = tf.add(tf.matmul(out1, weights['h2']), biases['b2'])
    out2 = tf.nn.relu(out2)
    out3 = tf.layers.dense(out2, 2)

    return out3

# Construct model
logits = neural_net(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)

    for step in range(1, num_steps+1):
        training = pd.concat([X_train,Y_train],axis=1).sample(batch_size)
        batch_x = pd.DataFrame(training,columns=list(X_train))
        batch_y = pd.DataFrame(training,columns=list(Y_train))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    t_data = X_test
    t_results = Y_test
    print("Testing Accuracy:",
        sess.run(accuracy, feed_dict={X: t_data,
                                      Y: t_results}))
