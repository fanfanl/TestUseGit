import tensorflow as tf
import numpy as np

###自定义模型，继承tf.Module

class SampleLayer(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.x = tf.Variable([[1.0, 3.0]], name="x_trainable")
        self.y = tf.Variable(2.0, trainable=False, name="y_non_trainable")

    def __call__(self, input):
        return self.x * input + self.y


simple_layer = SampleLayer(name="my_layer")

output = simple_layer(tf.constant(1.0))
print("output", output)
print("Layer_name", simple_layer.name)
print("Trainalbe Varibles: ", simple_layer.trainable_variables)


class Module(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        print("init")
        self.lay_1 = SampleLayer("layer_1")
        self.lay_2 = SampleLayer("layer_2")

    def __call__(self, x):
        print("call")
        x = self.lay_1(x)
        output = self.lay_2(x)
        return output


custorm_model = Module(name="model_name")

output = custorm_model(tf.constant(1.0))
print("output", output)
print("model_name: ", custorm_model.name)
print("Trainerable Variables: ", custorm_model.trainable_variables)


class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layers1 = tf.keras.layers.Dense(16, activation='relu')
        self.layers2 = tf.keras.layers.Dense(32, activation='relu')

    def call(self, inputs):
        x = self.layers1(inputs)
        out = self.layers2(x)
        return out


custorm_model = CustomModel()
output = custorm_model(tf.constant([[1.0, 2,0, 3.0]]))

print("output shape: ", output.shape)
print("Model name: ", custorm_model.name)

###使用keras构造模型, 可以使用tf.keras.Model定义模型，tf.keras.layers方便定义
class CustormModel(tf.keras.Model):
    def __init__(self):
        super(CustormModel, self).__init__()
        self.layer_1 = tf.keras.layers.Dense(16,activation=tf.nn.relu)
        self.layer_2 = tf.keras.layers.Dense(32, activation=None)

    def call(self, inputs):
        x = self.layer_1(inputs)
        out = self.layer_2(x)
        return out

custorm_model = CustormModel()

output = custorm_model(tf.constant([[1.0, 2.0, 3.0]]))

print("output shape: ",  output.shape)
print("Model name: ", custorm_model.name)
print(custorm_model.summary)