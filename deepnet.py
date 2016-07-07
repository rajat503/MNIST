import tensorflow as tf

#load dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#create session
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = weight_variable([3, 3, 1, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, 16, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)


W_conv3 = weight_variable([3, 3, 16, 32])
b_conv3 = bias_variable([32])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

W_conv4 = weight_variable([3, 3, 32, 64])
b_conv4 = bias_variable([64])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool4_flat = tf.reshape(h_pool4, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 1024])
b_fc2 = bias_variable([1024])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc4 = weight_variable([1024, 1024])
b_fc4 = bias_variable([1024])

h_fc4 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc4) + b_fc4)
h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)


W_fc3 = weight_variable([1024, 10])
b_fc3 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc4_drop, W_fc3) + b_fc3)

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)), reduction_indices=[1]))
    tf.scalar_summary('cross entropy', cross_entropy)

train_step1 = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
train_step2 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step3 = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc_summary = tf.scalar_summary('accuracy', accuracy)


merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('./train', sess.graph)
test_writer = tf.train.SummaryWriter('./test')

sess.run(tf.initialize_all_variables())

for i in range(16000):
    batch = mnist.train.next_batch(200)
    if i%100==0:
        summ = sess.run(acc_summary, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
        test_writer.add_summary(summ, i)
        # val_acc=accuracy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0}))
        # print("validation accuracy %g"%accuracy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0}))
        # train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        # print("step %d"%(i))
        # print("training accuracy %g"%(train_accuracy))
    if i%100 == 0 and i!=0:
        print "step", i, "loss", loss_val
    if i>=13000:
        _, loss_val, summary = sess.run([train_step3, cross_entropy, merged], feed_dict={x:batch[0], y_: batch[1], keep_prob: 0.5})
    if i>5000 and i<13000:
        _, loss_val, summary = sess.run([train_step2, cross_entropy, merged], feed_dict={x:batch[0], y_: batch[1], keep_prob: 0.5})
    if i<=5000:
        _, loss_val, summary = sess.run([train_step1, cross_entropy, merged], feed_dict={x:batch[0], y_: batch[1], keep_prob: 0.5})
    if i>100:
        train_writer.add_summary(summary, i)

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

det, pred = sess.run([tf.argmax(y_conv,1), correct_prediction], feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

import cv2
for i in range(len(pred)):
    if pred[i]==False:
        x=mnist.test.images[i]*255
        x=x.reshape(28,28)
        print "Predicted class", det[i]
        print "Correct class", mnist.test.labels[i]
        img = cv2.imshow("a",x)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
