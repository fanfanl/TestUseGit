# coding=utf-8

"""
@author: mengfan
@date: 2022/4/19 11:03 下午
@purpose: 识别mnist数据集
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST数据集相关参数
INPUT_NODE = 784 #输入层的节点数
OUTPUT_NODE = 10 #输出层的节点数，即分类的数目

# 配置神经网络的参数
LAYER1_NODE = 500 # 隐藏层的节点数
BATCH_SIZE = 100 # 一个batch中的训练数据个数

LEARNING_RATE_BASE = 0.8  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  #学习率的衰退率
REGULARIZATION_RATE = 0.0001 # 正则化的系数
TRAINING_STEPS = 30000 # 训练轮数
MOVING_AVERAGE_DECAY = 0.99 # 活动平均衰减率

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    """
    辅助函数，给定输入和所有的参数向前计算钱箱传播结果，支持计算参数平均值的类
    :param input_tensor: 输入tensor
    :param avg_class: 滑动平均模型，None表示不使用
    :param weights1:
    :param biases1:
    :param weight2:
    :param biases2:
    :return:
    """
    if avg_class == None:
        # 计算前向传播结果，隐藏层采用了relu激活函数
        layers = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layers, weights2) + biases2
    else:
        # 首先使用滑动平均函数avg_class.average函数计算变量的滑动平均值
        # 之后再计算相应的神经网络向前传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) +
                            avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

def train(mnist):
    """
    训练模型的过程
    :param mnist:
    :return:
    """
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 生成隐藏层的参数


