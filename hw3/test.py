import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None,None])
y = tf.placeholder(tf.float32, [None,None])
k = tf.reduce_max(y,axis=1)

sess = tf.Session()

y = sess.run(k, feed_dict={x:[[0,1],[1,0],[0,1]], y:[[1,2],[2,1],[3,4]]})

print(y)