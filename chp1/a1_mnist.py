#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a1_mnist.py
@Time: 2019-07-22 10:08
@Last_update: 2019-07-22 10:08
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
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))