from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# data_path = keras.utils.get_file('housing.data', "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data")
data_path = "housing.data"
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX',
                'RM', 'AGR', 'DIS', 'RAD', 'TAX', 'PTRATION', 'B', 'LSTAT', 'MEDV']

raw_dataset = pd.read_csv(data_path, names=column_names, sep=" ",
                          skipinitialspace=True)
dataset = raw_dataset.copy()

p = 0.8
trainDataset = dataset.sample(frac=p, random_state=0)
testDataset = dataset.drop(trainDataset.index)

train_input = trainDataset['RM']
train_target = trainDataset['MEDV']
test_input = testDataset['RM']
test_target = testDataset['MEDV']


# 在不制定激活函数的清华下a(x) = x，
def linear_model():
    model = keras.Sequential([layers.Dense(1, use_bias=True, input_shape=(1,), name='layer')])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-05, amsgrad=False, name="Adam"
    )

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model

model = linear_model()

model.summary()

n_epochs = 4000
batch_size = 256
n_idle_epochs = 100
n_epochs_log = 200
n_sample_save = n_epochs_log * train_input.shape[0]
print("Checkpoint is saved for each {} samples".format(n_sample_save))

# 停止训练的机制为在n_idle_epochs之后loss没有提升就停止训练
# 可以查看 https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=n_idle_epochs, min_delta=0.001)

# 创建一个经过特定次数输出日志的自定义callback
predictions_list = []
class NEPOCHLogger(tf.keras.callbacks.Callback):
    def __init__(self, per_epoch=100):
        '''
        display: 在输出loss之前等待的batches数目
        :param per_epoch:
        '''
        self.seen = 0
        self.per_epoch = per_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.per_epoch == 0:
            print("Epoch {}, loss {:.2f}, val_loss: {:.2f}, mae {:.2f}, val_mae: {:.2f}, mse {:.2f}, "
                  "val_mse {:.2f}".format(epoch, logs['loss'], logs['val_loss'], logs['mae'],
                                          logs['val_mae'], logs['mse'], logs['val_mse']))


log_display = NEPOCHLogger(per_epoch=n_epochs_log)

import os
checkpoint_path="training/cp-{epoch: 05d}.ckpt"
checkpoint_dis = os.path.dirname(checkpoint_path)

# 构建一个保存每5个epoch保存模型weights的callback
checkpointCallback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=n_sample_save
)

# 利用checkpoint_path保存参数
model.save_weights(filepath=checkpoint_path.format(epoch=0))

# 定义keras TensorBoard callback
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H:%M:%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

history = model.fit(
    train_input, train_target, batch_size=batch_size, epochs=n_epochs,
    validation_split=0.1, verbose=0, callbacks=[earlyStopping, log_display, tensorboard_callback,
                                                checkpointCallback]
)

print(type(history), "keys:", history.history.keys())

predictions = model.predict(test_input)
print(type(predictions), predictions)
predictions = predictions.flatten()
print("predictions.flatten()")
print(predictions)

