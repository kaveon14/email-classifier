import tensorflow as tf
from __future__ import absolute_import, division, print_function

tf.enable_eager_execution()

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))


a = tf.constant([[1, 2], [3, 4]])
print(a)

b = tf.add(a, 1)
print(b)

print(a * b)