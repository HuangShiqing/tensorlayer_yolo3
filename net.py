import tensorflow as tf
import tensorlayer as tl
import numpy as np

from tensorlayer.layers import *

from tensorlayer.deprecation import deprecated_alias
import tensorlayer._logging as logging

import logging as log


class AddLayer(Layer):
    @deprecated_alias(layer='prev_layer', end_support_version=1.9)
    def __init__(self, prev_layer=None, name='0'):
        super(AddLayer, self).__init__(prev_layer=prev_layer, name='add' + name)

        input_size = prev_layer.outputs.get_shape().as_list()[1]
        input_ch = prev_layer.outputs.get_shape().as_list()[3]

        # for route in routes:
        #     if abs(route) >= len(prev_layer.all_layers):
        #         raise Exception("beyond the num of layers")

        logging.info("RouteLayer  %s: routes:%s" % ('add' + name, str(-5)))

        self.inputs = prev_layer.outputs

        out = prev_layer.all_layers[-5]
        # for i, route in enumerate(routes):
        # out.append(prev_layer.all_layers[route])

        with tf.variable_scope('add' + name):
            # self.outputs = tf.concat([o for o in out], 3)
            self.outputs = out + self.inputs

        self.all_layers.append(self.outputs)

        print('    {} res     {}                {} x {} x {}   ->   {} x {} x {}'.format(name, str(int(name) - 3),
                                                                                         input_size,
                                                                                         input_size,
                                                                                         input_ch,
                                                                                         input_size, input_size,
                                                                                         input_ch))


def conv2d_unit(prev_layer, filters, kernels, strides=1, name='0'):
    input_size = prev_layer.outputs.get_shape().as_list()[1]
    input_ch = prev_layer.outputs.get_shape().as_list()[3]

    network = Conv2dLayer(
        prev_layer,
        act=tf.identity,
        shape=(kernels, kernels, input_ch, filters),
        strides=(1, strides, strides, 1),
        padding='SAME',
        name='conv' + name)
    network = BatchNormLayer(network, act=tf.nn.leaky_relu, is_train=True, name='bn' + name)

    out_size = network.outputs.get_shape().as_list()[1]
    out_ch = network.outputs.get_shape().as_list()[3]

    print('    {} conv     {}  {} x {} / {}   {} x {} x {}   ->   {} x {} x {}'.format(name, filters,
                                                                                       kernels, kernels,
                                                                                       strides, input_size, input_size,
                                                                                       input_ch,
                                                                                       out_size, out_size, out_ch))
    return network


def residual_block(inputs, filters, name='0'):
    x = conv2d_unit(inputs, filters, 1, name=name)
    x = conv2d_unit(x, 2 * filters, 3, name=str(int(name) + 1))
    x = AddLayer(x, name=str(int(name) + 2))

    return x


def stack_residual_block(inputs, filters, n, name='0'):
    """Stacked residual Block
    """
    x = residual_block(inputs, filters, name=name)

    for i in range(n - 1):
        x = residual_block(x, filters, name=str(int(name) + 3 * (i + 1)))

    return x


logger = log.getLogger('tensorlayer')
logger.setLevel(level=log.ERROR)

print('layer     filters    size              input                output')
input_pb = tf.placeholder(tf.float32, [None, 416, 416, 3])
net = InputLayer(input_pb, name='input')
net = conv2d_unit(net, 32, 3, name='0')

net = conv2d_unit(net, 64, 3, strides=2, name='1')
net = stack_residual_block(net, 32, 1, name='2')

net = conv2d_unit(net, 128, 3, strides=2, name='5')
net = stack_residual_block(net, 64, 2, name='6')

net = conv2d_unit(net, 256, 3, strides=2, name='12')
net = stack_residual_block(net, 128, 8, name='13')

net = conv2d_unit(net, 512, 3, strides=2, name='37')
net = stack_residual_block(net, 256, 8, name='38')

net = conv2d_unit(net, 1024, 3, strides=2, name='62')
net = stack_residual_block(net, 512, 4, name='63')

exit()


def darknet_53(x):
    network = InputLayer(x, name='input')
    network_1 = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 3, 32],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv1_1')
    network = Conv2dLayer(
        network_1,
        act=tf.nn.relu,
        shape=[3, 3, 32, 64],  # 64 features for each 3x3 patch
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='conv1_2')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[1, 1, 64, 32],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv1_3')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 32, 64],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv1_4')

    output = network.outputs + network_1.outputs

    network.print_layers()
    return output, network


# input_pb = tf.placeholder(tf.float32, [None, 416, 416, 3])
# out, net = darknet_53(input_pb)

print("ok")
