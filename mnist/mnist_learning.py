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
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算前向传播结果，这里并不计算滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的变量
    global_step = tf.Variable(0, trainable=False) # 不需要计算滑动平均值，trainable=False

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均，tf.trainable_variables返回图上集合的可训练变量
    # GraphKeys.TRAINABLE_VARIABLES中的元素
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均值的前向传播结果，滑动平均不会改变变量本身的值
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵刻画预测值和真实值之间的差距的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)

    # 总损失= 交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization
    # learning_rate_base: 基础学习率，global_step: 当前迭代的轮数，decay_steps: 过完所有的训练数据需要的迭代次数
    # learning_rate_decay: 学习率衰减速度
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY
    )

    # 优化损失函数，这里损失函数包含了交叉熵损失和L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 既需要进行反向传播更新神经网络中的参数，又需要更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 向前传播运算结果
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    # 该组上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32) )

    #初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 准备验证集，用于判断停止的条件
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }
        # 准备测试数据，训练部分是不可见的，只是作为模型优劣的最后评价标准
        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
        }

        # 迭代的训练神经网络
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                # 每1000轮输出一次在验证数据集上的测试结果
                validete_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy "
                      "using average model is %g "%(i, validete_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x:xs, y_: ys})

        # 训练结束之后， 在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy "
              "using average model is %g " % (TRAINING_STEPS, test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets("./data", one_hot=True)
    train(mnist)

# tf提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == "__main__":
    tf.app.run()

