import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
  @tf.function(jit_compile=True)
  def call(self, x1):
    return tf.cast(x1, dtype=tf.float64)


x1 = tf.constant([1.1, 2.1, 3.1], dtype=tf.float32)

m = Model()
expected_value = m(x1)

print(expected_value)