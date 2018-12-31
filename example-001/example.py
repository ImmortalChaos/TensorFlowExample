#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 200개의 점을 찍어 데이터 모델을 만들자 
number_of_points = 200

x_point = []
y_point = []
a = 0.22
b = 0.78

for i in range(number_of_points) :
	x = np.random.normal(0.0, 0.5)
	y = a*x + b + np.random.normal(0.0, 0.1)
	x_point.append([x])
	y_point.append([y])

plt.plot(x_point, y_point, 'o', label='Input Data')
plt.legend()
plt.show()

# TensorFlow를 이용해 A와 b를 다시 유추하기 위한 변수를 선언한다.
# 변수 A는 -1과 1 사이의 임의의 값으로 초기화했고, 변수 b는 0으로 초기화했다.
A = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = A * x_point + b

# 비용함수(cost_function)는 A와 B의 예측 값이 실제의 값과 어느 정도 차이가 있는지 연산한다.
# 비용함수는 예측이 얼마나 정확한지를 나타내며 에측의 평균값이 실제의 값으로부터 얼마나 떨어져 있는지의 분산을 나타낸다.
# 이 값이 작을수록 미지수 A와 B를 실제와 더 가깝게 예측했다고 볼 수 있다.
cost_function = tf.reduce_mean(tf.square(y - y_point))

# 경사 하강법(Gradient Descent)을 이용해 cost_function값을 최소화할 수 있다.
# 여기서 0.5는 경사 하강법의 학습률(learning rate)을 의미하며 최적의 값에 도달하기 위해 경사를 얼마나 급격하게 적용할지를 정의한다.
# 이 값이 매우 크다면 최적의 해를 지나칠 확률이 높고 너무 작다면 최솟값을 찾기 위해 너무 많은 순회를 반복해야 할 것이다.
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cost_function)

# 변수를 초기화한다.
model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	# 변수 A와 B의 값을 도출할 수 있게 세션을 통해 모델 학습을 20회 정도 반복한다.
	for step in range(0, 21) :
		session.run(train)
		if (step % 5) == 0:
			# 학습된 변수 A와 B를 이용한 방정식 y = Ax + b를 출력한다.
			plt.plot(x_point, y_point, 'o', 
				label='step = {}'
				.format(step))
			plt.plot(x_point, session.run(A) * x_point + session.run(b))
			plt.legend()
			plt.show()
