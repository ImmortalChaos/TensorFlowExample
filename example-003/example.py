#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#학습하기 위한 100개의 이미지를 리턴
train_pixels, train_list_values = mnist.train.next_batch(100)

test_pixels, test_list_of_values = mnist.test.next_batch(10)

#분류기를 제작하는 데 사용할 train_pixel_tensor와 test_pixel_tensor 텐서를 정의한다.
train_pixel_tensor = tf.placeholder("float", [None, 784])
test_pixel_tensor = tf.placeholder("float", [784])

distance = tf.reduce_sum(tf.abs(tf.add(train_pixel_tensor, tf.negative(test_pixel_tensor))), reduction_indices=1)
pred = tf.arg_min(distance, 0)

accuracy = 0.
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for i in range(len(test_list_of_values)):
		nn_index = sess.run(pred, feed_dict={train_pixel_tensor:train_pixels, test_pixel_tensor:test_pixels[i,:]})
		print( "Test N=", i, "Predicated Class: ", np.argmax(train_list_values[nn_index]), \
			"True Class: ", np.argmax(test_list_of_values[i]))
		if np.argmax(train_list_values[nn_index])==np.argmax(test_list_of_values[i]):
			accuracy += 1./len(test_pixels)
			print("Result = ", accuracy)

