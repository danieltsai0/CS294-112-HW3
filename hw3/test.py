import tensorflow as tf
import numpy as np


for i in range(10):
	towrite = str(i)+"\n"
	with open("test.txt","a+") as f:
		f.write(towrite)
# x = tf.placeholder(tf.float32, [2,3])
# y = tf.argmax(x, axis=0)

# sess = tf.Session()

# y = sess.run(y, feed_dict={x:[[1,2,3],[4,5,6]]})

# print(y)