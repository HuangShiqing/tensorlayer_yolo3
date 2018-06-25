import tensorflow as tf
import tensorlayer as tl
import numpy as np
import cv2

from tensorlayer.layers import *
from tensorflow.contrib import layers

img = cv2.imread('dog.jpg')
img = img[:, :, ::-1]  # RGB image
im_sized = cv2.resize(img, (416, 416))
im_sized = np.expand_dims(im_sized, axis=0)

weights_path = 'yolov3.weights'
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
    shape=(32,),  # (32,),
    dtype='float32',
    buffer=weights_file.read(32 * 4))
bn_weights = np.ndarray(
    shape=(3, 32),
    dtype='float32',
    buffer=weights_file.read(32 * 12))
bn_weight_list = [
    bn_weights[0],  # scale gamma
    conv_bias,  # shift beta
    bn_weights[1],  # running mean
    bn_weights[2]  # running var
]
weights_size = 32 * 3 * 3 * 3
conv_weights = np.ndarray(
    shape=[32, 3, 3, 3],  # [32, 3, 3, 3],
    dtype='float32',
    buffer=weights_file.read(weights_size * 4))
conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
conv_weights = conv_weights
weights_file.close()

input_pb = tf.placeholder(tf.float32, [None, 416, 416, 3])
net = InputLayer(input_pb, name='input')
net = Conv2dLayer(
    prev_layer=net,
    act=tf.identity,
    shape=(3, 3, 3, 32),
    strides=(1, 1, 1, 1),
    padding='SAME',
    b_init=None,
    name='conv_1',
    W_init_args={'regularizer': layers.l2_regularizer(5e-4)})
net = BatchNormLayer(net, epsilon=1e-3, act=lambda x: tl.act.lrelu(x, 0.1), is_train=False, name='bn_1')
net = UpSampling2dLayer(net, (2, 2), is_scale=True, method=1, name='upsample_1')

out = net.outputs

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    tensor_conv_w = tf.global_variables('conv_1')[0]
    tensor_bn_beta = tf.global_variables('bn_1')[0]
    tensor_bn_gamma = tf.global_variables('bn_1')[1]
    tensor_bn_mean = tf.global_variables('bn_1')[2]
    tensor_bn_variance = tf.global_variables('bn_1')[3]

    tf.assign(tensor_conv_w, conv_weights).eval()
    tf.assign(tensor_bn_beta, conv_bias).eval()
    tf.assign(tensor_bn_gamma, bn_weights[0]).eval()
    tf.assign(tensor_bn_mean, bn_weights[1]).eval()
    tf.assign(tensor_bn_variance, bn_weights[2]).eval()

    a = sess.run(out, feed_dict={input_pb: im_sized})

    exit()
