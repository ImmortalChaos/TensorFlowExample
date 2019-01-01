#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data

# MNIST를 이용하여 K-Means Clustering을 학습합니다.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


num_vectors = 1000
num_clusters = 4
num_steps = 100

x_values = []
y_values = []
vector_values = []

for i in range(num_vectors) :
	if np.random.random() > 0.5 :
		x_values.append(np.random.normal(0.4, 0.7))
		y_values.append(np.random.normal(0.2, 0.8))
	else:
		x_values.append(np.random.normal(0.6, 0.4))
		y_values.append(np.random.normal(0.8, 0.5))

vector_values = list(zip(x_values, y_values))
vectors = tf.constant(vector_values)

plt.plot(x_values, y_values, 'o', label='Input Data')
plt.legend()
plt.show()
