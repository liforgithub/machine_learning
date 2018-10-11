#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/11 0011 上午 9:34
# @Author  : 李雪洋
# @File    : 简单的卷积神经网络.py
# @Software: PyCharm
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# 卷积神经网络需要很多的权重和偏置，所以定义好初始化的函数以便使用
# 给权重制造一些随机的噪声来打破完全对称(本次设定为截断的正态分布噪声，标准差设置0.1)
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 由于使用ReLU,所以也给偏置增加一些小的正值(0.1)用来避免死亡节点
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积层和池化层需要重复使用,因此也需要定义成函数
# tf.nn.conv2d是tensorflow的2维卷积函数
# x是输入  W是卷积的参数(W ---> [5,5,1,32]:  5x5的卷积核， 1个通道， 32个卷积核，也就是提取32个特征值)
# strides是卷积模板移动的步长，都是1代表不遗漏图片上的每一个像素点， Padding 代表边界处理方式，SAME代表卷积的输入和输出尺寸一致
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# tf.nn.max_pool 是tensorflow中的最大最大池化函数，这里我们使用2x2的最大池化，即将一个2x2的像素块降为1x1的像素。最大池化会保留原始像素块中灰度值最高的那一个像素，即保留最显著的特征
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

# x是特征， y_是真实的label，因为卷积神经网络会利用空间结构信息,因此需要将1D的输入向量转为2D的图片结构,即从1x784的形式转为原始的28x28的结构. -1代表样本数量不固定,1代表颜色通道数量
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 定义第一个卷积层
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# conv2d先进行卷积操作,并加上偏置，再用ReLU激活函数进行非线性处理,最后在进行池化
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二个卷积层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层
# 经过了两次2x2的池化，边长变为原来的1/4,图片尺寸由28x28变为7x7，而第二个卷积层的卷积核数量为64个，所以其输出的tensor尺寸为7x7x64
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# 对第二个卷积层输出结果进行变形，将其转化为1D的向量,然后连接一个全连接层，隐含节点1024个，并使用ReLU激活
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了减轻过拟合,使用dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 将dropout层的输出连接一个softmax层，得到最后的概率输出
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    print(i)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy: %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
