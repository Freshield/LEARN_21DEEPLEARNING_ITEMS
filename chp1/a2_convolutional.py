#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a2_convolutional.py
@Time: 2019-07-22 14:52
@Last_update: 2019-07-22 14:52
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os
import numpy as np
import tensorflow as tf

data_dir = '/media/freshield/SSD_1T/Data/a8_21dl/chapter_1_data/MNIST_data'

mnist = input_data.read_data_sets(data_dir, one_hot=True)

print(mnist.train.images.shape)
print(mnist.train.labels.shape)

save_dir = os.path.join(data_dir, 'raw')
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

# for i in range(20):
#     image_array = mnist.train.images[i, :]
#
#     image_array = image_array.reshape(28, 28)
#
#     filename = os.path.join(save_dir, 'mnist_train_%d.jpg' % i)
#
#     # scipy.misc.toimage(image_array, cmin=0., cmax=1.).save(filename)
#     one_hot_label = mnist.train.labels[i, :]
#     label = np.argmax(one_hot_label)
#     print('%s: %d' % (filename, label))

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding='SAME')

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(50)

    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch_xs, y_: batch_ys, keep_prob: 1.0
        })
        print('step %d, training accuracy %g' % (i, train_accuracy))

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1}))