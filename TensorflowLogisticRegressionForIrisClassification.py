# using logistic regression for iris classification. using vanilla tensorflow.
# note: using random forest classifier would be probably easier and faster in this case,
# but I use tensorflow for practice

import tensorflow as tf
import numpy as np
import sklearn.datasets as sd
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split

#how many epochs
steps=200
#how many batches
batch_size=5
#1e-3 is little slow but precise
learning_rate=1e-3

#number of features/dimensions of input (iris features)
features=4
#number of classes (iris types)
classes=3

#changing categories from integers to one hot shape
def onehot(_y):
    lb = LabelBinarizer()
    lb.fit(range(max(_y)+1))
    return lb.transform(_y)

#load iris dataset. its toy dataset provided by scikitlearn
iris = sd.load_iris()
train_x, test_x, train_y, test_y = \
    train_test_split(iris.data, onehot(iris.target), test_size=0.4, random_state=0)

#print head of input and output
print("input:")
print(train_x[:3])
print("output:")
print(train_y[:3])

#placeholder for input and output
x = tf.placeholder("float", [None, features])
y = tf.placeholder("float", [None, classes])

#variables for weights and biases
w = tf.Variable( tf.zeros([features,classes]))
b = tf.Variable(tf.zeros([classes]))

#softmax activation function
prediction= tf.nn.softmax(tf.matmul(x,w)+b)
#cross entropy loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
#optimizer declaration
optimizer = tf.train.AdamOptimizer(learning_rate)
#minimaze loss with optimizer
minimize_loss = optimizer.minimize(loss)

#split data into batches
data_size=len(iris.target)
split_size=int(data_size/batch_size)
splitted_x=np.split(train_x, split_size)
splitted_y=np.split(train_y, split_size)

#initialize all variables
init = tf.global_variables_initializer()
with tf.Session() as s:
    #initialize session
    s.run(init)
    #train model
    for epoch in range(steps):
        #train in batches
       for i in range(split_size):
            #feed training data
            s.run(minimize_loss, feed_dict={x: splitted_x[i], y: splitted_y[i]})

    #check if the guess is correct
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    #get accuracy
    calculating_accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    #validate with both training and test sets
    print("Accuracy on training set:", calculating_accuracy.eval({x: train_x, y: train_y}))
    print("Accuracy on validation set:", calculating_accuracy.eval({x: test_x, y: test_y}))