import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.int32, [None])
act_t = tf.one_hot(x, depth=5, on_value=1.0, off_value=0.0, dtype=tf.float32, name="action_one_hot")
y = tf.placeholder(tf.float32, [4,5])
k = tf.reduce_max(tf.multiply(y, act_t), axis=1)

# ind = tf.transpose(tf.stack([tf.to_int32(tf.range(y.get_shape()[0])),x]))
# k = tf.gather_nd(y,ind)

sess = tf.Session()

y = sess.run(k, feed_dict={x:[0,1,2,3], y:[[-1,2,3,4,0],[-5,6,-7,8,9],[9,10,-11,12,-21],[13,-14,15,-16,0]]})

print(y)