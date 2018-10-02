import pandas as pd
from matplotlib import pyplot
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Model architecture parameters
n_stocks = 500
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1

data = pd.read_csv('stock_100.csv')
data = data.drop(['DATE'], 1)
n = data.shape[0]
p = data.shape[1]
data = data.values

train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

a = tf.placeholder(dtype=tf.int8)
b = tf.placeholder(dtype=tf.int8)
c = tf.add(a, b)
graph = tf.Session()
graph.run(c, feed_dict={a: 5, b: 4})
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

