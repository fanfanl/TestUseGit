import tensorflow as tf
import numpy as np
import timeit

### propose: 学习使用tf图结构与无图结构的区别
class ModelShallow(tf.keras.Model):
    def __init__(self):
        super(ModelShallow, self).__init__()
        self.dense1 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(20, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(30, activation=tf.nn.softmax)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense2(x)
        out = self.dense3(x)
        return out


class ModelDeep(tf.keras.Model):
    def __init__(self):
        super(ModelDeep, self).__init__()
        self.dense1 = tf.keras.layers.Dense(1000, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(2000, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(3000, activation=tf.nn.softmax)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense2(x)
        out = self.dense3(x)
        return out


model_shallow_with_eager = ModelShallow()
model_shallow_with_graph = tf.function(ModelShallow())
model_deep_with_eager = ModelDeep()
model_deep_with_graph = tf.function(ModelDeep())

sample_input = tf.random.uniform([60, 28, 28])


print(timeit.timeit(lambda: model_shallow_with_eager(sample_input), number=1000))
print(timeit.timeit(lambda: model_shallow_with_graph(sample_input), number=1000))
print(timeit.timeit(lambda: model_deep_with_eager(sample_input), number=1000))
print(timeit.timeit(lambda: model_deep_with_graph(sample_input), number=1000))


"""
输入结果为：
2.5442089770000003
1.2375989120000002

"""