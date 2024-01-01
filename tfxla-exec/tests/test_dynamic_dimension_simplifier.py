import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()


  def call(self, x1):
    x2 = tf.reshape(x1, [2,2])
    return tf.reshape(x2, [4])

m = Model()

input_shape = [4]
x1 = tf.constant([4.,5.,6.,7.], shape=input_shape)

y = m(x1)

print(y)
#print(m.call.experimental_get_compiler_ir(x1)(stage='hlo'))
#print(m.call.experimental_get_compiler_ir(x1)(stage='optimized_hlo'))