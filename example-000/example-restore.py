#-*- coding: utf-8 -*-
import tensorflow as tf

weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")

# 위에 정의한 변수를 변형하여 다른 이름으로 재 정의 할 수 있다
w2 = tf.Variable(weights.initialized_value(), name="w2")
w_twice = tf.Variable(weights.initialized_value() * 2.0, name="w_twice")

val1 = tf.constant(0.0)
val2 = tf.constant(0.0)
result = tf.add(val1, val2)

print("weight : " + str(weights.initialized_value()))
print("biases : " + str(biases.initialized_value()))
print("w2 : " + str(w2.initialized_value()))
print("w_twice : " + str(w_twice.initialized_value()))
print("value 1 : " + str(val1))
print("value 2 : " + str(val2))
print("result : " + str(result))

saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess, "saved/model.ckpt")
	print("Model restored.")

	# 값을 찍어 보면 탠소 플로우 베리어블 객체의 내용을 확인 가능
	# 더하기 연산 수행과 그 결과
	print("Result :", sess.run(result))
