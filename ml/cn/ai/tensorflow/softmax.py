import tensorflow as tf
from tensorflow import graph_util
from tensorflow.examples.tutorials.mnist import input_data
import time
import os
import datetime
from tensorflow.core.framework import summary_pb2

mnist = input_data.read_data_sets('D:\python_workspace\ml_file\MNIST_data', one_hot=True)

tf.placeholder(tf.float32, [None, 784])

x = tf.placeholder(tf.float32, [None, 784], name="pic_input")
W = tf.Variable(tf.random_uniform([784, 10]), name="w")
b = tf.Variable(tf.random_uniform([1, 10]), name="b")
op = tf.matmul(x, W)+b


y_ = tf.placeholder(tf.float32, [None, 10])

y = tf.nn.softmax(op, name="softmax")
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

kk = tf.nn.softmax_cross_entropy_with_logits(logits=op, labels=y_)

print(kk)

cross_entropy = tf.reduce_sum(kk)
print(cross_entropy)



timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))

# Summaries for loss and accuracy
loss_summary = tf.summary.scalar("loss", cross_entropy)
acc_summary = tf.summary.scalar("accuracy", accuracy)
test_loss_summary = tf.summary.scalar("test_loss", cross_entropy)
test_acc_summary = tf.summary.scalar("test_accuracy", accuracy)

train_summary_dir = os.path.join(out_dir, "summaries", "train")

global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.GradientDescentOptimizer(0.0001)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

sess = tf.Session()


train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)



grad_summaries = []
for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
grad_summaries_merged = tf.summary.merge(grad_summaries)

train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
test_summary_op = tf.summary.merge([test_loss_summary, test_acc_summary])

init = tf.global_variables_initializer()


sess.run(init)

i = 0
for i in range(800):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  _, step , summaries, entropy, acc = sess.run([train_op, global_step, train_summary_op, cross_entropy, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
  time_str = datetime.datetime.now().isoformat()
  print("{}: step {}, entropy {:g}, acc {:g}".format(time_str, step, entropy, acc))
  test_summary_op_result, acc_test = sess.run([test_summary_op, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
  #summ = summary_pb2.Summary()
  print("{}: step {}, test acc {:g}".format(time_str, step, acc_test))
  #print("summaries:"+str(summ.ParseFromString(summaries)))
  train_summary_writer.add_summary(summaries, step)
  train_summary_writer.add_summary(test_summary_op_result, step)
#

saver=tf.train.Saver(max_to_keep=3)
saver.save(sess, 'ckpt/mnist.ckpt', global_step=100)
saver.save(sess, 'ckpt/mnist.ckpt', global_step=101)

model_file=tf.train.latest_checkpoint('ckpt/')
saver.restore(sess, model_file)
val_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print('final test acc {:g}'.format(val_acc))


builder = tf.saved_model.builder.SavedModelBuilder('./model4java')
builder.add_meta_graph_and_variables(sess, ["mytag"])
builder.save()


#test_one_x = mnist.test.images[0:1, :];
#test_one_y = mnist.test.labels[0:1]

#print(sess.run(tf.argmax(y, 1), feed_dict={x:test_one_x}))

#print(sess.run(tf.argmax(y, 1), feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

train_summary_writer.close()





