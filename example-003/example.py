#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST를 이용하여 KNN을 학습합니다.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 학습하기 위한 100개의 이미지를 사용한다.
train_pixels, train_list_values = mnist.train.next_batch(100)
# 이미지 10개에 대해서 작성한 알고리즘을 테스트한다.
test_pixels, test_list_of_values = mnist.test.next_batch(10)

# 분류기를 제작하는 데 사용할 train_pixel_tensor와 test_pixel_tensor 텐서를 정의한다.
train_pixel_tensor = tf.placeholder("float", [None, 784])
test_pixel_tensor = tf.placeholder("float", [784])

# 비용함수는 픽셀 간의 거리에 해당한다.
# tf.reduce_sum : 텐서의 차원들을 탐색하며 개체들의 총 합을 계산한다. 
distance = tf.reduce_sum(tf.abs(tf.add(train_pixel_tensor, tf.negative(test_pixel_tensor))), reduction_indices=1)
# 거리 함수를 최소화하기 위해 arg_min을 사용하여 가장 작은 거리를 갖는 인덱스(최근접 이웃)을 리턴하게 한다.
pred = tf.arg_min(distance, 0)

# 정확도(accuracy)는 분류기의 최종 결과를 계산하는 데 사용하는 매개변수
accuracy = 0.
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for i in range(len(test_list_of_values)):
		# 앞서 정의한 pred 함수를 사용해 최근접 이웃 인덱스를 평가한다.
		nn_index = sess.run(pred, feed_dict={train_pixel_tensor:train_pixels, test_pixel_tensor:test_pixels[i,:]})
		# Predicated Class :  예측한 클래스
		# True Class : 실제 클래스
		print( "Test N=", i, "Predicated Class: ", np.argmax(train_list_values[nn_index]), \
			"True Class: ", np.argmax(test_list_of_values[i]))
		# 분류기에 대한 정확도를 평가한다.
		if np.argmax(train_list_values[nn_index])==np.argmax(test_list_of_values[i]):
			accuracy += 1./len(test_pixels)
		print("Result(accuracy) = ", accuracy)

