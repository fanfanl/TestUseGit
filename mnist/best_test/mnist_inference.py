# coding=utf-8

"""
@author: mengfan
@date: 2022/5/5 11:02 下午
@purpose:定义模型向前传播的过程
"""
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

# 通过tf.get_variable 函数来获取变量。在训练神经网络时会创建这些变量；在测试时可以通过保存的模型加载这些变量的取值
# 更加方便的是，可以在加载变量的过程中可以通过字典的方式重新加载不同命名方式的变量
# 因此在训练的时候使用变量自身，在测试时使用变量的滑动平均值

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1)
    )

    # 当给出了正则化生成函数时，将当前变量的正则化损失加入名字为losses的集合
    # 通过add_to_collection函数将变量加入到losses的自定义集合
    if regularizer != None:
        tf.add_to_collection("losses", regularizer(weights))
    return weights

# 定义神经网络的向前传播过程
def inference(input_tensor, regularizer):
    # 声明第一层神经网络的变量并完成前向传播过程
    with tf.variable_scope("layer1"):
        weights = get_weight_variable(
            [INPUT_NODE, LAYER1_NODE], regularizer
        )
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    with tf.variable_scope("layer2"):
        weights = get_weight_variable(
            [LAYER1_NODE, OUTPUT_NODE], regularizer
        )
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2

