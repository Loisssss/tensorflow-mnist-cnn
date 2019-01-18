import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 定义卷积和池化函数,两边两个值默认为1，中间两个1代表分别从x方向以及y方向的步长
def conv2d(input, filter):
    return tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')

def max_pooling(input):
    return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#load mnist dataset
mnist_dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)

#定义图片的占位符，是一个Tensor，shape为[None, 784]，即可以传入任意数量的
# 图片，每张图片的长度为784(28 * 28)
x = tf.placeholder(tf.float32, [None, 784])
#定义图片的占位符， 是一个Tensor，shape为【None， 10】，即可以传入任意数量的
#图片，每张图片的长度为10
y_ = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

#---------------------------conv1--------------------------
#5为卷积核的高度和宽度
#1为输入channel的数量（灰度图）； 32为输出channel的数量，即为32个滤波器，提取32个特征
w_conv1 = weight_variable([5,5,1,32]) #定义滤波器
b_conv1 = bias_variable([32])   #每个滤波器对应一个bias
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pooling(h_conv1)

#----------------------------conv2---------------------------
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pooling(h_conv2)

#----------------------------fc1 layer------------------------
# 全连接层，一共有1024个神经元
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# h_pool2的shape为[batch,7,7,64],将其reshape为[-1, 7*7*64], 三维变一维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, 0.5) #dropout 防止过拟合

#----------------------------fc2 layer------------------------
w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)


#---------------------training and evaluation------------
def compute_accuracy(v_x, v_y):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x:v_x})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy,feed_dict={x: v_x, y_: v_y})
    return result
#计算loss, 来衡量模型的误差
# cross_entroy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(prediction), reduction_indices=[1]))
cross_entroy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=prediction))

#开始反向传播,用Adam优化器来训练模型，使得loss最小
train_step = tf.train.AdamOptimizer(learning_rate=(1e-4)).minimize(cross_entroy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#训练1000次
#每次从mnist_dataset中取出100张图片进行训练，
#把这100张图片的像素点存在batch_x变量里，把图片代表的0-9数字存在batch_y变量里
for i in range(100):
    batch_x,batch_y = mnist_dataset.train.next_batch(10)
    sess.run(train_step,feed_dict={x: batch_x, y_: batch_y})
    if i % 10 == 0:
        print(compute_accuracy(mnist_dataset.test.images, mnist_dataset.test.labels))