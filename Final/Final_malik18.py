"""
This program is to analyze and test to see how videogames would sell based on  Rating, Genre, Platform, Publisher,etc
"""

import os
import sys

#Savng all outputs and redirecting to surpress unneeded output
stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.stderr = stderr
np.random.seed(42)
TESTING = True

dataset = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv",keep_default_na=True)
 
dataset = dataset.loc[dataset.Year_of_Release < 2017] #Potential in 2017 not good since it only accounts for January
dataset.Year_of_Release = dataset['Year_of_Release'].astype(int) #Convert Years to Int
dataset = dataset[(dataset['Platform'] == 'PS3') | (dataset['Platform'] == 'PS4') | (dataset['Platform'] == 'X360') | (dataset['Platform'] == 'XOne') | (dataset['Platform'] == 'Wii') | (dataset['Platform'] == 'WiiU') | (dataset['Platform'] == 'PC')]
dataset = dataset.dropna(subset=['Critic_Score'])
dataset['Rating'] = dataset['Rating'].fillna(dataset['Rating'].mode()[0])
dataset['Year_of_Release'] = dataset['Year_of_Release'].fillna(dataset['Year_of_Release'].median())
dataset.replace('tbd',np.NaN,inplace=True,regex=True)
dataset['User_Score'] = dataset['User_Score'].fillna(dataset['User_Score'].median())
dataset['User_Count'] = dataset['User_Count'].fillna(dataset['User_Count'].median())
dataset = dataset.drop(['Name', 'NA_Sales', 'Publisher','Developer','EU_Sales', 'JP_Sales', 'Other_Sales'], axis=1)
dataset = pd.get_dummies(dataset, columns=['Platform','Genre','Rating'])

Y = dataset[['Global_Sales']]
del dataset['Global_Sales']
X = dataset

num_input = len(list(X))
num_classes = 1

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

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None,num_classes])

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
    out3 = tf.add(tf.matmul(out2, weights['h3']), biases['b3'])
    out3 = tf.nn.relu(out3)
    out4 = tf.add(tf.matmul(out3, weights['h4']), biases['b4'])
    out4 = tf.nn.relu(out4) 
    out5 = tf.layers.dense(out4, 2)

    return out5

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
            if TESTING:                                                    
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                     "{:.3f}".format(acc))

    if TESTING:
        print("Optimization Finished!")

    if TESTING:
        print("Testing Accuracy:",
            sess.run(accuracy, feed_dict={X: X_test,Y: Y_test}))
    else:
        print(sess.run(accuracy, feed_dict={X: X_test,Y: Y_test}))

