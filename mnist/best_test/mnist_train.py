# coding=utf-8

"""
@author: mengfan
@date: 2022/5/5 11:24 下午
@purpose: 定义数字识别问题的训练过程
"""
import os
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import tensorflow as tf
# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

#模型保存的路径和文件名
MODEL_SAVE_PATH = "./model"
MODEL_NAME = "model.ckpt"

def train(mnist):
    x = tf.placeholder(tf.float32, shape=[None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, shape=[None, mnist_inference.OUTPUT_NODE], name='y-output')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    # 直接使用mnist_inference中定义的前向传播过程
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)


    # 定义损失函数、学习率、滑动平均操作以及训练过程
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    print(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化tf持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: xs, y_: ys})
            # 每1000轮保存一次模型
            if i % 1000 == 0:
                # 输出当前的训练情况，这里只输出模型在当前训练batch上的损失函数大小。
                # 在验证集上的正确率信息会有一个单独的程序生成
                print("After %d training step(s), loss on training batch"
                      " is %g." % (step, loss_value))
                # 保存当前模型，参数中有global_step参数，可以让每个保存模型的文件名末
                # 加上训练的轮数，比如"model.ckpt-1000"表示训练1000轮之后的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                           global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()





