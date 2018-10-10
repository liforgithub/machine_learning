#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/10 0010 下午 15:21
# @Author  : 李雪洋
# @File    : 多层感知机.py
# @Software: PyCharm

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

# 输入节点数
in_units = 784
# 第一级隐藏层节点数
h1_units = 300

# W1 b1为隐藏层的权重和偏置，我们将偏置全部赋值为0，权重通过truncated_normal函数初始化为截面的正态分布，其标准差为0.1
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))

# 输出层的权重和偏置
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
# 因为训练和预测时，Dropout的比率keep_prob(保留节点的概率)是不一样的,通常在训练时小于1，而预测的时候则等于1，所以也把Dropout的比率作为计算图的输入，定义成一个placeholder
keep_prob = tf.placeholder(tf.float32)

# 实现一个激活函数ReLU的隐藏层 ReLU  --->   y=relu(Wx + b)
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
# 调用dropout随机将一部分节点置为0， 这里的keep_prob参数即为保留数据而不置为0的比例没在训练时应该是小于1的，用以制造随机性，防止过拟合；在预测时应该等于1，即使用全部特征来预测样本的类型
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
# 输出层
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

y_ = tf.placeholder(tf.float32, [None, 10])
# 定义交叉信息熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 利用自适应学习速率优化器adagrad
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

tf.global_variables_initializer().run()

# 训练步骤， 加入了keep_prob作为计算图的输入， 并且在训练时设为0.75， 即保留75%的节点，其余的25%置为0.一般来说，对越复杂越大规模的神经网络，Dropout效果更好
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
