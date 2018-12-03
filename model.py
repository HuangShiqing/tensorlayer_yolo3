import tensorflow as tf
from tensorlayer.layers import *
from net import Gb_all_layer_out, ResLayer, RouteLayer, upsample, conv2d_unit, detection

import numpy as np
import time
import cv2


def inference(input, n_class, is_train=True):
    # n_class = 80
    net = InputLayer(input, name='input')
    net = conv2d_unit(net, filters=32, kernels=3, strides=1, bn=True, is_train=is_train, name='0')
    net = conv2d_unit(net, filters=64, kernels=3, strides=2, bn=True, is_train=is_train, name='1')
    net = conv2d_unit(net, filters=32, kernels=1, strides=1, bn=True, is_train=is_train, name='2')
    net = conv2d_unit(net, filters=64, kernels=3, strides=1, bn=True, is_train=is_train, name='3')
    net = ResLayer(net, res=1, name='4')
    net = conv2d_unit(net, filters=128, kernels=3, strides=2, bn=True, is_train=is_train, name='5')
    net = conv2d_unit(net, filters=64, kernels=1, strides=1, bn=True, is_train=is_train, name='6')
    net = conv2d_unit(net, filters=128, kernels=3, strides=1, bn=True, is_train=is_train, name='7')
    net = ResLayer(net, res=5, name='8')
    net = conv2d_unit(net, filters=64, kernels=1, strides=1, bn=True, is_train=is_train, name='9')
    net = conv2d_unit(net, filters=128, kernels=3, strides=1, bn=True, is_train=is_train, name='10')
    net = ResLayer(net, res=8, name='11')
    net = conv2d_unit(net, filters=256, kernels=3, strides=2, bn=True, is_train=is_train, name='12')
    net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, is_train=is_train, name='13')
    net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, is_train=is_train, name='14')
    net = ResLayer(net, res=12, name='15')
    net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, is_train=is_train, name='16')
    net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, is_train=is_train, name='17')
    net = ResLayer(net, res=15, name='18')
    net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, is_train=is_train, name='19')
    net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, is_train=is_train, name='20')
    net = ResLayer(net, res=18, name='21')
    net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, is_train=is_train, name='22')
    net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, is_train=is_train, name='23')
    net = ResLayer(net, res=21, name='24')
    net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, is_train=is_train, name='25')
    net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, is_train=is_train, name='26')
    net = ResLayer(net, res=24, name='27')
    net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, is_train=is_train, name='28')
    net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, is_train=is_train, name='29')
    net = ResLayer(net, res=27, name='30')
    net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, is_train=is_train, name='31')
    net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, is_train=is_train, name='32')
    net = ResLayer(net, res=30, name='33')
    net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, is_train=is_train, name='34')
    net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, is_train=is_train, name='35')
    net = ResLayer(net, res=33, name='36')
    net = conv2d_unit(net, filters=512, kernels=3, strides=2, bn=True, is_train=is_train, name='37')
    net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, is_train=is_train, name='38')
    net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, is_train=is_train, name='39')
    net = ResLayer(net, res=37, name='40')
    net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, is_train=is_train, name='41')
    net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, is_train=is_train, name='42')
    net = ResLayer(net, res=40, name='43')
    net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, is_train=is_train, name='44')
    net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, is_train=is_train, name='45')
    net = ResLayer(net, res=43, name='46')
    net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, is_train=is_train, name='47')
    net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, is_train=is_train, name='48')
    net = ResLayer(net, res=46, name='49')
    net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, is_train=is_train, name='50')
    net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, is_train=is_train, name='51')
    net = ResLayer(net, res=49, name='52')
    net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, is_train=is_train, name='53')
    net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, is_train=is_train, name='54')
    net = ResLayer(net, res=52, name='55')
    net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, is_train=is_train, name='56')
    net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, is_train=is_train, name='57')
    net = ResLayer(net, res=55, name='58')
    net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, is_train=is_train, name='59')
    net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, is_train=is_train, name='60')
    net = ResLayer(net, res=58, name='61')
    net = conv2d_unit(net, filters=1024, kernels=3, strides=2, bn=True, is_train=is_train, name='62')
    net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, is_train=is_train, name='63')
    net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, is_train=is_train, name='64')
    net = ResLayer(net, res=62, name='65')
    net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, is_train=is_train, name='66')
    net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, is_train=is_train, name='67')
    net = ResLayer(net, res=65, name='68')
    net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, is_train=is_train, name='69')
    net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, is_train=is_train, name='70')
    net = ResLayer(net, res=68, name='71')
    net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, is_train=is_train, name='72')
    net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, is_train=is_train, name='73')
    net = ResLayer(net, res=71, name='74')
    net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, is_train=is_train, name='75')
    net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, is_train=is_train, name='76')
    net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, is_train=is_train, name='77')
    net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, is_train=is_train, name='78')
    net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, is_train=is_train, name='79')
    net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, is_train=is_train, name='80')
    net = conv2d_unit(net, filters=3 * (5 + n_class), kernels=1, strides=1, act='liner', bn=False, is_train=is_train,
                      name='81')
    detection(net, '82')
    net = RouteLayer(net, [79], name='83')
    net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, is_train=is_train, name='84')
    net = upsample(net, scale=2, name='85')
    net = RouteLayer(net, [85, 61], name='86')
    net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, is_train=is_train, name='87')
    net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, is_train=is_train, name='88')
    net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, is_train=is_train, name='89')
    net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, is_train=is_train, name='90')
    net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, is_train=is_train, name='91')
    net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, is_train=is_train, name='92')
    net = conv2d_unit(net, filters=3 * (5 + n_class), kernels=1, strides=1, act='liner', bn=False, is_train=is_train,
                      name='93')
    detection(net, '94')
    net = RouteLayer(net, [91], name='95')
    net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, is_train=is_train, name='96')
    net = upsample(net, scale=2, name='97')
    net = RouteLayer(net, [97, 36], name='98')
    net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, is_train=is_train, name='99')
    net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, is_train=is_train, name='100')
    net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, is_train=is_train, name='101')
    net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, is_train=is_train, name='102')
    net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, is_train=is_train, name='103')
    net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, is_train=is_train, name='104')
    net = conv2d_unit(net, filters=3 * (5 + n_class), kernels=1, strides=1, act='liner', bn=False, is_train=is_train,
                      name='105')
    detection(net, '106')
    net_out = [Gb_all_layer_out[106], Gb_all_layer_out[94], Gb_all_layer_out[82]]

    return net_out


if __name__ == '__main__':
    input_pb = tf.placeholder(tf.float32, [None, 416, 416, 3])
    with tf.Session() as sess:
        out = infenence(input_pb)

        sess.run(tf.global_variables_initializer())
        for i in range(10):
            img = cv2.imread('4.jpg')
            img = cv2.resize(img, (416, 416))
            img = np.expand_dims(img, axis=0)

            time_1 = time.time()
            a = sess.run(out, feed_dict={input_pb: img})
            print(time.time() - time_1)
