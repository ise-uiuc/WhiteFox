
import tensorflow
class Model(tensorflow.keras.Model):
    def __init__(self):
        super().__init__()
    def call(self, x1, x2):
        v1 = x1[:, 0:2]
        v2 = x1[:, 2:4]
        v3 = x1[:, 4:7]
        v4 = x2[:, 2:6]
        v5 = x2[:, 0:3]
        v6 = x2[:, 6:9]
        v7 = x2[:, 3:7]
        v8 = v1 + v3
        v9 = v4 + v5
        v10 = v6 + v7
        v11 = v8 + v9
        v12 = v10 + v11
        v13 = v12 + v9
        v14 = v13[-1, :-1]
        v15 = v13[-1, -1]
        return v13, v6
# Inputs to the model
x1 = tf.random.uniform([1, 9], minval=0, maxval=2, dtype=tf.float64)
x2 = tf.random.uniform([1, 9], minval=0, maxval=2, dtype=tf.float64)
