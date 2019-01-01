#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data

def display_partition(x_values, y_values, assignment_values):
	labels = []
	# 각 클러스터에 대한 색을 지정
	colors = ["red", "blue", "green", "yellow"]
	for i in range(len(assignment_values)):
		labels.append(colors[(assignment_values[i])])
	color = labels
	df = pd.DataFrame(dict(x=x_values, y=y_values, color=labels))
	fig, ax = plt.subplots()
	ax.scatter(df['x'], df['y'], c=df['color'])
	plt.show()


# MNIST를 이용하여 K-Means Clustering을 학습합니다.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 군집화하고자 하는 점의 총 개수
num_vectors = 2000

# 군집화할 파티션 개수
num_clusters = 4

# k-means의 계산 반복 횟수
num_steps = 1000

# 초기 입력 데이터 구조체
x_values = []
y_values = []
vector_values = []

# 랜덤한 점들로 이루어진 학습 데이터 집합을 생성한다.
for i in range(num_vectors) :
	if np.random.random() > 0.5 :
		x_values.append(np.random.normal(0.4, 0.7))
		y_values.append(np.random.normal(0.2, 0.8))
	else:
		x_values.append(np.random.normal(0.6, 0.4))
		y_values.append(np.random.normal(0.8, 0.5))

# python의 zip함수를 사용해서 vector_values의 전체 목록을 얻는다.
vector_values = list(zip(x_values, y_values))
# verctor_values를 텐서플로에서 사용 가능한 상수로 변환한다.
vectors = tf.constant(vector_values)
# random_shffle을 사용해 인덱스를 지정한다.
n_samples = tf.shape(vector_values)[0]
random_indices = tf.random_shuffle(tf.range(0, n_samples))

begin = [0,]
size = [num_clusters,]
size[0] = num_clusters

# 각 인덱스들에 대한 초기 센트로이드를 갖는다.
centroid_indices = tf.slice(random_indices, begin, size)
centroids = tf.Variable(tf.gather(vector_values, centroid_indices))

# 앞에서 정의한 텐서들인 vectors와 centroids를 처리하기 위해 두 인자값에 지정된 크기로 확장해주는 텐서플로 함수 expand_dims를 사용한다.
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

# tf.subtract 함수로 차이를 구하기 위해 두 텐서의 형태를 표준화해준다.
vectors_subtration = tf.subtract(expanded_vectors, expanded_centroids)

# 텐서의 각 차원들에 대해 개체들의 합을 계산하는 tf.reduce_sum 함수와 
# vectors_subtration의 각 원소들의 제곱을 계산하는 tf.square 함수를 이용해 
# 유클리디언 거리에 대한 비용 함수를 작성한다.
euclidean_distances = tf.reduce_sum(tf.square(vectors_subtration), 2)
# 할당은 텐서 간의 가장 짧은 거리를 갖는 인덱스 값으로 지정
assignments = tf.to_int32(tf.argmin(euclidean_distances, 0))

# 새로운 그룹으로 묶는다.
partitions = tf.dynamic_partition(vectors, assignments, num_clusters)
# 센트로이드를 업데이트 한다.
# 하나의 그룹에 대해 reduce_mean을 실행해 군집의 평균을 찾고 새로운 센트로이드로 지정한다.
for partition in partitions:
	update_centroids = tf.concat(tf.expand_dims(tf.reduce_mean(partition, 0), 0), 0)

# 테스트와 알고리즘을 평가 시작.
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

for step in range(num_steps):
	_, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

display_partition(x_values, y_values, assignment_values)
plt.plot(x_values, y_values, 'o', label='Input Data')
plt.legend()
plt.show()
