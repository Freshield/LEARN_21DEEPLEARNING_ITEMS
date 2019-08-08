#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a1_test_input.py
@Time: 2019-08-08 10:50
@Last_update: 2019-08-08 10:50
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import os
import tensorflow as tf

file_dir = 'data/reader'
with tf.Session() as sess:

    filename_list = ['A.jpg', 'B.jpg', 'C.jpg']
    filename_list = [os.path.join(file_dir, name) for name in filename_list]

    filename_queue = tf.train.string_input_producer(
        filename_list, shuffle=True, num_epochs=5)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    tf.local_variables_initializer().run()

    threads = tf.train.start_queue_runners(sess=sess)

    i = 0
    while True:
        i += 1
        image_data = sess.run(value)

        with open('data/saver2/test_%d.jpg' % i, 'wb') as f:
            f.write(image_data)
