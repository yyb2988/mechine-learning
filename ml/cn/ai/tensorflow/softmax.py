import tensorflow as tf
from tensorflow import graph_util
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('D:\python_workspace\ml_file\MNIST_data', one_hot=True)

tf.placeholder(tf.float32, [None, 784])

x = tf.placeholder(tf.float32, [None, 784], name="pic_input")
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([1, 10]))
op = tf.matmul(x, W)+b


y_ = tf.placeholder(tf.float32, [None, 10])

y = tf.nn.softmax(op, name="softmax")
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=op, labels=y_))


train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

i = 0
for i in range(10000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print('final:', sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

test_one_x = mnist.test.images[0:1, :];
test_one_y = mnist.test.labels[0:1]

print(sess.run(tf.argmax(y, 1), feed_dict={x:test_one_x}))

print(sess.run(tf.argmax(y, 1), feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["softmax"])
with tf.gfile.FastGFile('MNIST_softmax.pb', mode='wb') as f:
  f.write(constant_graph.SerializeToString())



