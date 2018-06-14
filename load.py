import tensorflow as tf
import tensorlayer as tl
import numpy as np

from tensorlayer.layers import *

checkpoint_dir = 'D:/DeepLearning/code/tensorlayer_yolo3/ckpt/'

weights_path = 'yolov3.weights'
config_path = 'yolov3.cfg'
# Load weights and config.
print('Loading weights.')
weights_file = open(weights_path, 'rb')
major, minor, revision = np.ndarray(
    shape=(3,), dtype='int32', buffer=weights_file.read(12))
if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
    seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
else:
    seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
print('Weights Header: ', major, minor, revision, seen)

conv_bias = np.ndarray(
    shape=(32,),
    dtype='float32',
    buffer=weights_file.read(32 * 4))
bn_weights = np.ndarray(
    shape=(3, 32),
    dtype='float32',
    buffer=weights_file.read(32 * 12))

weights_size = np.product([3, 3, 3, 32])
conv_weights = np.ndarray(
    shape=[32, 3, 3, 3],
    dtype='float32',
    buffer=weights_file.read(weights_size * 4))
conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])

is_train = True
with tf.variable_scope('conv0'):
    b = tf.get_variable(
        name='b_conv2d', shape=(32,), dtype=tf.float32, initializer=tf.zeros_initializer()
    )
    W = tf.get_variable(
        name='W_conv2d', shape=[3, 3, 3, 32], dtype=tf.float32, initializer=tf.zeros_initializer()
    )
with tf.variable_scope('bn0'):
    gamma = tf.get_variable(
        'gamma', shape=(32,), initializer=tf.zeros_initializer(), dtype=tf.float32, trainable=is_train,
    )
    beta = tf.get_variable(
        'beta', shape=(32,), initializer=tf.zeros_initializer(), dtype=tf.float32, trainable=is_train
    )
    moving_mean = tf.get_variable(
        'moving_mean', shape=(32,), initializer=tf.zeros_initializer(), dtype=tf.float32, trainable=False
    )
    moving_variance = tf.get_variable(
        'moving_variance', shape=(32,), initializer=tf.constant_initializer(1.), dtype=tf.float32, trainable=False,
    )

saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    b = tf.assign(b, conv_bias)
    W = tf.assign(W, conv_weights)
    gamma = tf.assign(gamma, bn_weights[0])
    beta = tf.assign(beta, conv_bias)
    moving_mean = tf.assign(moving_mean, bn_weights[1])
    moving_variance = tf.assign(moving_variance, bn_weights[2])

    sess.run(b)
    sess.run(W)
    sess.run(gamma)
    sess.run(beta)
    sess.run(moving_mean)
    sess.run(moving_variance)

    saver.save(sess, checkpoint_dir + "model.ckpt")

#load
input_pb = tf.placeholder(tf.float32, [None, 416, 416, 3])
network = InputLayer(input_pb, name='input')
network = Conv2dLayer(
    network,
    act=tf.identity,
    shape=(3, 3, 3, 32),
    strides=(1, 1, 1, 1),
    padding='SAME',
    name='conv' + '0')
network = BatchNormLayer(network, act=tf.nn.leaky_relu, decay=0.99, epsilon=1e-3, is_train=True, name='bn' + '0')
# 读取ckpt里保存的参数
# sess = tf.InteractiveSession()
saver = tf.train.Saver()
# 如果有checkpoint这个文件可以加下面这句话，如果只有一个ckpt文件就不需要这个if
if tf.train.get_checkpoint_state(checkpoint_dir):  # 确认是否存在
    saver.restore(sess, checkpoint_dir + "model.ckpt")
    print("load ok!")
else:
    print("ckpt文件不存在")
