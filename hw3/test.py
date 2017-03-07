import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.int32, [None])
y = tf.placeholder(tf.float32, [3,3])
z = tf.one_hot(x,depth=3)

k = tf.reduce_max(y,axis=1)

sess = tf.Session()

y = sess.run(k, feed_dict={x:[0,1,0], y:[[1,2,3],[4,5,6],[7,8,9]]})

print(y)