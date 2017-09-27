import tensorflow as tf
import numpy as np

batch_size=1
batch = tf.placeholder(tf.float32, [None, 14, 14, 32])
img = np.random.rand(batch_size, 14, 14, 32)
maxpooled = tf.nn.max_pool(batch, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
with tf.Session() as sess:
    output = sess.run(maxpooled, feed_dict={batch:img})
    print(batch.shape, output.shape)