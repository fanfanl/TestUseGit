# coding=utf-8

"""
@author: mengfan
@date: 2022/5/7 7:02 下午
@purpose:测试程序
"""
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train
# 每10s加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, shape=[None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, shape=[None, mnist_inference.OUTPUT_NODE], name='y-output')

        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        # 直接通过封装好的函数计算前向传播的结果，正则化函数设置为None
        y = mnist_inference.inference(x, None)

        # 使用前向传播的结果可以计算正确率，
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平均值。
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        # 每隔EVAL_INTERVAL_SECDS秒调用一次计算正确率的过程以检测训练过程中正确率的变化
        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名获取模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split("/")[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)

                    print("After %s training step(s), validation accuracy = %g."%(global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(EVAL_INTERVAL_SECS)
def main(argv=None):
    mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
    evaluate(mnist)

if __name__ == "__main__":
    tf.app.run()