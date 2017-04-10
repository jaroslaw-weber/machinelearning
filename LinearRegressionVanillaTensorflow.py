import tensorflow as tf
import numpy as np

#define train data. function is f(x)=2x+1. we are going to predict this function with linear regression
train_x = np.asarray ([1,2,3,4,5,6,7,8,9])
train_y = [i*2+1 for i in train_x]

#after 800 steps error gets to 2e-3
steps=800
#learning rate. it is common to use .001 and it works nicely here
learning_rate=1e-3

#placeholder for input and output
x = tf.placeholder("float")
y = tf.placeholder("float")

#weight and bias. in this case we use only one weight and one bias
w = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

#get prediction
prediction=tf.add(tf.mul(x,w),b)
#calculate loss based on prediction. use mse for loss function
loss = tf.reduce_mean(tf.pow(y - prediction,2))
#declare optimizer.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#minimize loss wit declared optimizer
minimize_loss = optimizer.minimize(loss)

#in tensorflow we need to initilize all variables before session
init = tf.global_variables_initializer()

# we run session. weight and bias should coverge to w=2, b=1
with tf.Session() as s:
    #initialize session
    s.run(init)
    #train model
    for epoch in range(steps):
        for (_x, _y) in zip(train_x, train_y):
            s.run(minimize_loss, feed_dict={x: _x, y: _y})
    #show loss after training
    final_error=s.run(loss, feed_dict={x:train_x, y:train_y})
    print("error after training : {0}".format(final_error))
    #show weights
    print ("weight={0} bias={1}".format(s.run(w), s.run(b)))
    #test model
    def test_model(test_x):
        test_prediction= test_x* s.run(w)+s.run(b)
        print("for input : {0}, model predicted {1}".format(test_x,test_prediction ))
        return
    test_model(11)
    test_model(20)
    #you could also use cross validation to make sure the prediction is right


