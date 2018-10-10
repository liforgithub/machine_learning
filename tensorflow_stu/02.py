#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/10 0010 上午 11:08
# @Author  : 李雪洋
# @File    : 02.py
# @Software: PyCharm

# 第一个简单的例子,使用tensorflow实现了一个简单的机器学习算法softmax Regression,这可以算作是一个没有隐藏层的最浅的神经网络。
# 步骤分为四步
# 1. 定义算法公式，也就是神经网络forward时的计算
# 2. 定义loss，选定优化器，并指定优化器优化loss
# 3. 迭代地对数据进行训练
# 4. 在测试集或验证集上对准确率进行评估

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 将session注册为默认的session,之后的运算也默认的运行在这个session里面
sess = tf.InteractiveSession()

# placeholder是占位符， 即输入数据的地方，placeholder第一个参数是输入参数的参数类型， 第二个参数是tensor的shape,也就是数据的尺寸，这里的none代表不限条数的输入，784代表的是每条输入都是一个784维的向量
x = tf.placeholder(tf.float32, [None, 784])

# Variable是用来存储模型参数的，在模型的迭代中是持久存在的，它可以长期存在并且可以在每轮的迭代中被更新
# 初始化权重和偏执为0
# 权重  784维 10代表10类， one-hot编码是10维的向量(其实就是有多少种判定结果)
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 实现Softmax Regression算法: y=softmax(Wx+b)
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义信息熵(cross-entropy) 判断模型对真实概率分布估计的准确程度
y_ = tf.placeholder(tf.float32, [None, 10])

# reduce_mean 用来对每个betch数据结果求均值， reduce_sum是求和
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 使用SGD进行优化
# 设定的学习速率为0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 执行tensorflow全局参数初始化器
tf.global_variables_initializer().run()

# 执行迭代训练操作,每次随机从训练集中抽取100条样本构成一个mini-batch,并feed给placeholder，然后调用train_step对这些样本进行训练
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

# 对模型进行准确性验证
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 我们统计全部样本预测的accuracy，这里需要先用tf.cast将之前correct_prediction输出的bool值转换成float32，再求平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
