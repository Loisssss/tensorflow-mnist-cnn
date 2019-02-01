import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np # linear algebra
from sklearn.model_selection import train_test_split
import matplotlib.cm as cm


train_file = "./kaggleData/train.csv"
test_file = "./kaggleData/test.csv"
output_file = "./kaggleData.submission.csv"

def dense_to_one_hot(labels_dense, num_calsses):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_calsses
    labels_one_hot = np.zeros((num_labels, num_calsses))
    #flat返回的是一个迭代器
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

# load the data
train_dataset = pd.read_csv(train_file)
print(train_dataset.shape)

# test_dataset = pd.read_csv(test_file)

Y_train_all = train_dataset["label"].values.ravel()
Y_train_all = dense_to_one_hot(Y_train_all, 10)
print(Y_train_all.shape)


# Drop the label colum in train dataset
X_train_all = train_dataset.drop(labels="label", axis=1)
print("图片大小为：", str(X_train_all.shape[1]) + " pixels")
X_train_all = X_train_all.values.reshape(-1, 28, 28, 1).astype("float32")
print("图片长度和高度为： " + str(X_train_all.shape[1]))

X_train, X_test, Y_train, Y_test = train_test_split(X_train_all, Y_train_all, test_size=0.2)
print("X_train shape", X_train.shape)
print("Y_train shape", Y_train.shape)
print("X_test shape", X_test.shape)
print("Y_test shape", Y_test.shape)


# display image
def display(img):
    # (784) => (28,28)
    one_image = img.reshape(28,28)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
# display(X_train[10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, name='Weights')
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, name='Bias')
    return tf.Variable(initial)

# 定义卷积和池化函数,两边两个值默认为1，中间两个1代表分别从x方向以及y方向的步长
def conv2d(input, filter):
    return tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')

def max_pooling(input):
    return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


x = tf.placeholder("float", [None, 28, 28, 1])
y_ = tf.placeholder("float", [None,10])

#---------------------------conv1--------------------------
#5为卷积核的高度和宽度
#1为输入channel的数量（灰度图）； 32为输出channel的数量，即为32个滤波器，提取32个特征
w_conv1 = weight_variable([5,5,1,32]) #定义滤波器
b_conv1 = bias_variable([32])   #每个滤波器对应一个bias
h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
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
with tf.name_scope("Prediction_softmax"):
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name='prediction')

#(小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(prediction,b,name='logits_eval')

with tf.name_scope("Accuracy"):
    with tf.name_scope("Correct_Prediction"):
        # 完成训练后，对模型的准确率进行验证
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(prediction, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.name_scope("Loss"):
    cross_entroy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=prediction))

with tf.name_scope("TrainStep_AdamOptimizer"):
    train_step = tf.train.AdamOptimizer(learning_rate=(1e-4)).minimize(cross_entroy)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
batch_size = 100
n_batch = int (len(X_train) / batch_size)
for i in range(32):
    for batch in range(n_batch):
        batch_x = X_train[batch*batch_size:(batch+1)*batch_size]
        batch_y = Y_train[batch*batch_size:(batch+1)*batch_size]
        sess.run(train_step,feed_dict={x: batch_x, y_: batch_y})
    if i % 1 == 0:
        # print("Test dataset accuracy: " + str(accuracy(mnist_dataset.test.images, mnist_dataset.test.labels)))
        print(str(i) +" Test dataset accuracy: " + str(sess.run(accuracy, feed_dict={x: X_test, y_: Y_test})))
