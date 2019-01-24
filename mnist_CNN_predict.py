import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from skimage import io, transform
import numpy as np

mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)
pb_path = "model/model.pb"
#导入pb文件到grraph中
with tf.gfile.FastGFile(pb_path, "rb") as f:
    #创建一个空的图
    graph_def = tf.GraphDef()
    #加载模型
    graph_def.ParseFromString(f.read())
    #复制定义好的计算图到默认图中
    _ = tf.import_graph_def(graph_def, name="")

with tf.Session() as sess:
    #获取输入tensor
    x = tf.get_default_graph().get_tensor_by_name("x_input:0")
    #获取预测tensor
    prediction = tf.get_default_graph().get_tensor_by_name("logits_eval：0")
    #取第100张图片测试
    one_image = np.reshape(mnist_data.test.images[100], [-1, 784])
    #将测试图片传入cnn中，zuochuinference
    out  = sess.run(prediction, feed_dict={x:one_image})
    prediction_label = np.argmax(out, 1)
    print("prediction_label= ", prediction_label)
    print("true label: ", np.argmax(mnist_data.test.labels[100], 0))