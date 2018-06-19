import tensorflow as tf
import tensorlayer as tl
import numpy as np

from tensorlayer.layers import *
from tensorflow.contrib import layers

from tensorlayer.deprecation import deprecated_alias
import tensorlayer._logging as logging

import logging as log


class ResLayer(Layer):
    @deprecated_alias(layer='prev_layer', end_support_version=1.9)
    def __init__(self, prev_layer=None, name='0'):
        super(ResLayer, self).__init__(prev_layer=prev_layer, name='add' + name)

        input_size = prev_layer.outputs.get_shape().as_list()[1]
        input_ch = prev_layer.outputs.get_shape().as_list()[3]

        # for route in routes:
        #     if abs(route) >= len(prev_layer.all_layers):
        #         raise Exception("beyond the num of layers")

        logging.info("ResLayer  %s: routes:%s" % ('add' + name, str(-5)))

        self.inputs = prev_layer.outputs

        out = prev_layer.all_layers[-5]
        # for i, route in enumerate(routes):
        # out.append(prev_layer.all_layers[route])

        with tf.variable_scope('res' + name):
            # self.outputs = tf.concat([o for o in out], 3)
            self.outputs = out + self.inputs

        self.all_layers.append(self.outputs)

        print('   {:3} res   {:2}                   {:3} x {:3} x {:4}   ->   {:3} x {:3} x {:4}'.format(name, str(
            int(name) - 3),
                                                                                                         input_size,
                                                                                                         input_size,
                                                                                                         input_ch,
                                                                                                         input_size,
                                                                                                         input_size,
                                                                                                         input_ch))


class RouteLayer(Layer):
    @deprecated_alias(layer='prev_layer', end_support_version=1.9)
    def __init__(self, prev_layer=None, routes=None, name='0'):
        super(RouteLayer, self).__init__(prev_layer=prev_layer, name='route' + name)

        if len(routes) == 1:
            self.inputs = tl.layers.get_layers_with_name(prev_layer, routes[0])[0]
        elif len(routes) == 2:
            self.inputs = tf.concat([tl.layers.get_layers_with_name(prev_layer, routes[0])[0],
                                     tl.layers.get_layers_with_name(prev_layer, routes[1])[0]], axis=-1)
            # self.inputs = tl.layers.get_layers_with_name(prev_layer, 'conv61')

        with tf.variable_scope('route' + name):
            # self.outputs = tf.concat([o for o in out], 3)
            self.outputs = self.inputs

        self.all_layers.append(self.outputs)

        print('   {:3} route   {}'.format(name, routes))
        # for i, route in enumerate(routes):
        #     out.append(prev_layer.all_layers[route])


def upsample(prev_layer, scale, name='0'):
    input_size = prev_layer.outputs.get_shape().as_list()[1]
    input_ch = prev_layer.outputs.get_shape().as_list()[3]

    out_net = UpSampling2dLayer(prev_layer, (scale, scale), is_scale=True, name='upsample' + name)

    out_size = out_net.outputs.get_shape().as_list()[1]
    out_ch = out_net.outputs.get_shape().as_list()[3]

    print('   {:3} upsample           {:4}x   {:3} x {:3} x {:4}   ->   {:3} x {:3} x {:4}'.format(name, scale,
                                                                                                   input_size,
                                                                                                   input_size,
                                                                                                   input_ch,
                                                                                                   out_size,
                                                                                                   out_size,
                                                                                                   out_ch))
    return out_net


def conv2d_unit(prev_layer, filters, kernels, strides=1, name='0', bn=True):
    input_size = prev_layer.outputs.get_shape().as_list()[1]
    input_ch = prev_layer.outputs.get_shape().as_list()[3]

    if strides > 1:
        network = ZeroPad2d(prev_layer, padding=((1, 0), (1, 0)), name='pad' + name)

    network = Conv2dLayer(
        prev_layer=prev_layer if strides is 1 else network,
        act=tf.identity if bn is True else lambda x: tl.act.lrelu(x, 0.1),  # tf.nn.leaky_relu,
        shape=(kernels, kernels, input_ch, filters),
        strides=(1, strides, strides, 1),
        padding='SAME' if strides is 1 else 'VALID',
        b_init=None if bn is True else tf.constant_initializer(value=0.0),
        name='conv' + name,
        W_init_args={'regularizer': layers.l2_regularizer(5e-4)})
    if bn is True:
        network = BatchNormLayer(network, act=lambda x: tl.act.lrelu(x, 0.1), is_train=True, name='bn' + name)

    out_size = network.outputs.get_shape().as_list()[1]
    out_ch = network.outputs.get_shape().as_list()[3]

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


def residual_block(inputs, filters, name='0'):
    x = conv2d_unit(inputs, filters, 1, name=name)
    x = conv2d_unit(x, 2 * filters, 3, name=str(int(name) + 1))
    x = ResLayer(x, name=str(int(name) + 2))

    return x


def stack_residual_block(inputs, filters, n, name='0'):
    """Stacked residual Block
    """
    x = residual_block(inputs, filters, name=name)

    for i in range(n - 1):
        x = residual_block(x, filters, name=str(int(name) + 3 * (i + 1)))

    return x

# n_class = 80
# logger = log.getLogger('tensorlayer')
# logger.setLevel(level=log.ERROR)
#
# print('layer     filters    size              input                output')
# input_pb = tf.placeholder(tf.float32, [None, 416, 416, 3])
# net = InputLayer(input_pb, name='input')
# net = conv2d_unit(net, 32, 3, name='0')
#
# net = conv2d_unit(net, 64, 3, strides=2, name='1')
# net = stack_residual_block(net, 32, 1, name='2')
#
# net = conv2d_unit(net, 128, 3, strides=2, name='5')
# net = stack_residual_block(net, 64, 2, name='6')
#
# net = conv2d_unit(net, 256, 3, strides=2, name='12')
# net = stack_residual_block(net, 128, 8, name='13')
#
# net = conv2d_unit(net, 512, 3, strides=2, name='37')
# net = stack_residual_block(net, 256, 8, name='38')
#
# net = conv2d_unit(net, 1024, 3, strides=2, name='62')
# net = stack_residual_block(net, 512, 4, name='63')
#
# net = conv2d_unit(net, 512, 1, name='75')
# net = conv2d_unit(net, 1024, 3, name='76')
# net = conv2d_unit(net, 512, 1, name='77')
# net = conv2d_unit(net, 1024, 3, name='78')
# net = conv2d_unit(net, 512, 1, name='79')
# net = conv2d_unit(net, 1024, 3, name='80')
# pred_yolo_1 = conv2d_unit(net, 3 * (5 + n_class), 1, name='81', bn=False).outputs
# net = RouteLayer(net, routes=['conv79'], name='83')
# net = conv2d_unit(net, 256, 1, name='84')
# net = upsample(net, scale=2, name='85')
# net = RouteLayer(net, routes=['upsample85', 'res61'], name='86')
#
# net = conv2d_unit(net, 256, 1, name='87')
# net = conv2d_unit(net, 512, 3, name='88')
# net = conv2d_unit(net, 256, 1, name='89')
# net = conv2d_unit(net, 512, 3, name='90')
# net = conv2d_unit(net, 256, 1, name='91')
# net = conv2d_unit(net, 512, 3, name='92')
# pred_yolo_2 = conv2d_unit(net, 3 * (5 + n_class), 1, name='93', bn=False).outputs
# net = RouteLayer(net, routes=['conv91'], name='95')
# net = conv2d_unit(net, 128, 1, name='96')
# net = upsample(net, scale=2, name='97')
# net = RouteLayer(net, routes=['upsample97', 'res36'], name='98')
#
# net = conv2d_unit(net, 128, 1, name='99')
# net = conv2d_unit(net, 256, 3, name='100')
# net = conv2d_unit(net, 128, 1, name='101')
# net = conv2d_unit(net, 256, 3, name='102')
# net = conv2d_unit(net, 128, 1, name='103')
# net = conv2d_unit(net, 256, 3, name='104')
# pred_yolo_3 = conv2d_unit(net, 3 * (5 + n_class), 1, name='105', bn=False).outputs
# exit()
#
# print("ok")
