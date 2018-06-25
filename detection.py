import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorflow.contrib import layers
# from net import ResLayer, RouteLayer, upsample, conv2d_unit, residual_block, stack_residual_block
from tensorlayer.deprecation import deprecated_alias

import cv2
import numpy as np

import logging as log

Gb_all_layer_out = list()


def conv2d_unit(prev_layer, filters, kernels, strides=1, name='0', bn=True):
    input_size = prev_layer.outputs.get_shape().as_list()[1]
    input_ch = prev_layer.outputs.get_shape().as_list()[3]

    if strides > 1:
        network = ZeroPad2d(prev_layer, padding=((1, 0), (1, 0)), name='layer_' + name + '_pad')

    network = Conv2dLayer(
        prev_layer=prev_layer if strides is 1 else network,
        act=tf.identity if bn is True else lambda x: tl.act.lrelu(x, 0.1),  # tf.nn.leaky_relu,
        shape=(kernels, kernels, input_ch, filters),
        strides=(1, strides, strides, 1),
        padding='SAME' if strides is 1 else 'VALID',
        b_init=None if bn is True else tf.constant_initializer(value=0.0),
        name='layer_' + name + '_conv',
        W_init_args={'regularizer': layers.l2_regularizer(5e-4)})
    if bn is True:
        network = BatchNormLayer(network, act=lambda x: tl.act.lrelu(x, 0.1), is_train=True,
                                 name='layer_' + name + '_bn')

    out_size = network.outputs.get_shape().as_list()[1]
    out_ch = network.outputs.get_shape().as_list()[3]

    Gb_all_layer_out.append(network.outputs)
    print('   {:3} conv     {:4}  {} x {} / {}   {:3} x {:3} x {:4}   ->   {:3} x {:3} x {:4}'.format(name, filters,
                                                                                                      kernels, kernels,
                                                                                                      strides,
                                                                                                      input_size,
                                                                                                      input_size,
                                                                                                      input_ch,
                                                                                                      out_size,
                                                                                                      out_size,
                                                                                                      out_ch))
    return network


class ResLayer(Layer):
    @deprecated_alias(layer='prev_layer', end_support_version=1.9)
    def __init__(self, prev_layer=None, name='0', res=None):
        super(ResLayer, self).__init__(prev_layer=prev_layer, name='layer_' + name + '_res')

        input_size = prev_layer.outputs.get_shape().as_list()[1]
        input_ch = prev_layer.outputs.get_shape().as_list()[3]

        self.inputs = prev_layer.outputs

        out = Gb_all_layer_out[res]

        with tf.variable_scope('res' + name):
            # self.outputs = tf.concat([o for o in out], 3)
            self.outputs = out + self.inputs

        self.all_layers.append(self.outputs)
        Gb_all_layer_out.append(self.outputs)
        print('   {:3} res   {:2}                   {:3} x {:3} x {:4}   ->   {:3} x {:3} x {:4}'.format(name, str(res),
                                                                                                         input_size,
                                                                                                         input_size,
                                                                                                         input_ch,
                                                                                                         input_size,
                                                                                                         input_size,
                                                                                                         input_ch))


logger = log.getLogger('tensorlayer')
logger.setLevel(level=log.ERROR)

checkpoint_dir = './ckpt/'
is_train = True
n_class = 80
input_pb = tf.placeholder(tf.float32, [None, 416, 416, 3])
net = InputLayer(input_pb, name='input')
net = conv2d_unit(net, filters=32, kernels=3, strides=1, name='0')
net = conv2d_unit(net, filters=64, kernels=3, strides=2, name='1')
net = conv2d_unit(net, filters=32, kernels=1, strides=1, name='2')
net = conv2d_unit(net, filters=64, kernels=3, strides=1, name='3')
net = ResLayer(net, res=1, name='4')
net = conv2d_unit(net, filters=128, kernels=3, strides=2, name='5')
net = conv2d_unit(net, filters=64, kernels=1, strides=1, name='6')
net = conv2d_unit(net, filters=128, kernels=3, strides=1, name='7')
net = ResLayer(net, res=5, name='8')
net = conv2d_unit(net, filters=64, kernels=1, strides=1, name='9')
net = conv2d_unit(net, filters=128, kernels=3, strides=1, name='10')
net = ResLayer(net, res=8, name='11')
net = conv2d_unit(net, filters=256, kernels=3, strides=2, name='12')
net = conv2d_unit(net, filters=128, kernels=1, strides=1, name='13')
net = conv2d_unit(net, filters=256, kernels=3, strides=1, name='14')
net = ResLayer(net, res=12, name='15')
net = conv2d_unit(net, filters=128, kernels=1, strides=1, name='16')
net = conv2d_unit(net, filters=256, kernels=3, strides=1, name='17')
net = ResLayer(net, res=15, name='18')
net = conv2d_unit(net, filters=128, kernels=1, strides=1, name='19')
net = conv2d_unit(net, filters=256, kernels=3, strides=1, name='20')
net = ResLayer(net, res=18, name='21')
net = conv2d_unit(net, filters=128, kernels=1, strides=1, name='22')
net = conv2d_unit(net, filters=256, kernels=3, strides=1, name='23')
net = ResLayer(net, res=21, name='24')




exit()
