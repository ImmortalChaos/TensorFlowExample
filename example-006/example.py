#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 다중 계층 퍼셉트론구조를 이용한 MNIST를 예측하는 예제
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 신경망의 학습률을 정한다.
learning_rate = 0.001

training_epochs = 20

batch_size = 100
display_step = 1

# 첫 번째 계층과 두 번째 계층의 뉴런수를 정의
n_hidden_1 = 256
n_hidden_2 = 256

# MNIST 입력 데이터(28 x 28)
n_input = 784

# 출력 클래스의 tn (0-9 숫자)
n_classes = 10

# 입력 계층은 x 텐서 [1 x 784]로 분류할 이미지를 나타낸다.
x = tf.placeholder("float", [None, n_input])

# 출력 텐서 y
y = tf.placeholder("float", [None, n_classes])

# 중간에 2개의 은닉 계층을 갖는다.
# 첫 번재 계층은 가중치 텐서인 h로 구성되고 [784 x 256]의 크기를 갖는데. 이 256 값은 계층의 총 노드 수를 나타낸다.
h = tf.Variable(tf.random_normal([n_input, n_hidden_1]))

# 계층 1에 해당하는 편향 텐서를 정의
bias_layer_1 = tf.Variable(tf.random_normal([n_hidden_1]))

# 계층 1
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,h), bias_layer_1))

# 두 번재 중간 계층은 [256 x 256] 모양의 가중치 텐서로 구성한다.
w = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))

# 계층 2에 해당하는 편향 텐서를 정의
bias_layer_2 = tf.Variable(tf.random_normal([n_hidden_2]))

# 계층 2
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w), bias_layer_2))

# 출력 계층의 가중치
output = tf.Variable(tf.random_normal([n_hidden_2, n_classes]))

# 출력 계층의 편향
bias_output = tf.Variable(tf.random_normal([n_classes]))
# 출력 계층은 계층 2로부터 입력 값으로 256개의 신호를 받게 되는데, 이 값은 각 숫자(0~9)에 대한 클래스에 속할 확률로 변환된다.
output_layer = tf.matmul(layer_2, output) + bias_output

# 로지스틱 회귀를 위해 비용 함수를 정의한다.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

# 비용 함수를 최소화할 옵티마이저
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 그래프 설정 값
avg_set = []
epoch_set = []

# 변수 초기화
init = tf.global_variables_initializer()
with tf.Session() as sess :
	sess.run(init)

	# 학습 사이클
	for epoch in range(training_epochs) :
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)
		# 모든 배치에 대해 반복
		for i in range(total_batch) :
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			# 배치 데이터를 사용해 학습한다.
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
			# 평균 비용 계산
			avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
		# 반복 단계마다 로그를 출력 
		if epoch % display_step == 0 :
			print ("Epoch:", '%04d'%(epoch+1), "cost=", "{:.9f}".format(avg_cost))
		avg_set.append(avg_cost)
		epoch_set.append(epoch+1)
	print("Training Phase finished")

	plt.plot(epoch_set, avg_set, 'o', label='MLP Training phase')
	plt.ylabel('cost')
	plt.xlabel('epoch')
	plt.legend()
	plt.show()

	# 모델 평가
	correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))

	# 정화도 평가
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print ("Model Accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))

