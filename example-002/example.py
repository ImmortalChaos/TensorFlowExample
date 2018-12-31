#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist_images = input_data.read_data_sets("MNIST_data/", one_hot=False)

pixels, real_values = mnist_images.train.next_batch(10)
print("list of values loaded ", real_values)

example_to_visualize = 5

image = pixels[example_to_visualize,:]
image = np.reshape(image, [28, 28])
plt.imshow(image)
plt.show()