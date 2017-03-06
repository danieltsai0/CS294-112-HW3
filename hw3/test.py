import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [2,3])
y = tf.argmax(x, axis=0)

sess = tf.Session()

y = sess.run(y, feed_dict={x:[[1,2,3],[4,5,6]]})

print(y)