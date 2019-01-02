#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 로지스틱 회귀 학습을 통해 MNIST를 예측하는 예제
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 트레닝을 위한 총 반복 횟수
training_epochs = 25
learning_rate = 0.01
batch_size = 100
display_step = 1

# 입력 텐서 x는 28 x 28 = 784픽셀 크기를 갖는 MNIST 데이터 이미지를 담는다.
x = tf.placeholder("float", [None, 784])
# 0-9사이의 숫자를 넣을 10개의 클래스
y = tf.placeholder("float", [None, 10])
# 모델의 가중치와 편향
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 각 이미지에 확률을 부여하기 위한 모델은 softmax로 한다.
# 
activation = tf.nn.softmax(tf.matmul(x, W) + b)

# 크로스 엔트로피로 나타낸 에러 최소화
cross_entropy = y*tf.log(activation)
# 비용함수
cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy, reduction_indices=1))
# 경사 하강법을 이용해 비용을 최소하한다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 그래프를 그릴 데이터 셋
avg_set = []
epoch_set = []

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	# 학습을 수행한다.
	for epoch in range(training_epochs) :
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)
		for i in range(total_batch) :
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			# 배치 데이터를 사용해 트레이닝한다.
			sess.run(optimizer, feed_dict={x: batch_xs, y:batch_ys})
			# 평균 비용 계산
			avg_cost += sess.run(cost, feed_dict={x: batch_xs, y:batch_ys})/total_batch
		# 각 반복 단계마다 로그 출력
		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
		avg_set.append(avg_cost)
		epoch_set.append(epoch+1)		
	print("Training phase finished")

	# 로지스틱 회귀 학습 과정을 그래프로 그린다.
	plt.plot(epoch_set, avg_set, 'o', label='Logistic Regression Training phase')
	plt.ylabel('cost')
	plt.xlabel('epoch')
	plt.legend()
	plt.show()

	# 모델 평가
	correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
	# 정확도를 계산한다.
	# 가장 높은 y값을 갖는 인덱스와 실제 숫자 벡터가 같다면 올바르게 예측한 것이다.
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print("MODEL accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))
