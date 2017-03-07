import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.int32, [None])
y = tf.placeholder(tf.float32, [None,None])
k = tf.reduce_max(tf.multiply(y,tf.one_hot(x,depth=4)),axis=1)

sess = tf.Session()

y = sess.run(k, feed_dict={x:[0,1,2,3], y:[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]})

print(y)