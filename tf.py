import tensorflow as tf

from get_data import Data

# 获取训练集和测试集
data = Data(100)
x_train, y_train, x_test, y_test = data.get_data_set()
x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())

# 第一层LSTM输入
lstm_1_input = tf.placeholder(dtype=tf.float16, shape=())
