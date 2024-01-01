import tensorflow as tf
import numpy as np
class Model(tf.keras.Model):

  def call(self, x):
    return tf.math.top_k(x, 1)

# Initializing the model
input_shape = [5, 4]
m = Model()

# Input to the model
x = tf.constant([[1, 5, 2, 4], [3, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16],
                 [17, 18, 19, 20]], shape=input_shape)


print(m(x))