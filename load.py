import tensorflow as tf
import tensorlayer as tl
import numpy as np

from tensorlayer.layers import *

from tqdm import tqdm
from net import *

logger = log.getLogger('tensorlayer')
logger.setLevel(level=log.ERROR)

checkpoint_dir = './ckpt/'

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

# conv_bias = np.ndarray(
#     shape=(32,),
#     dtype='float32',
#     buffer=weights_file.read(32 * 4))
# bn_weights = np.ndarray(
#     shape=(3, 32),
#     dtype='float32',
#     buffer=weights_file.read(32 * 12))
#
# weights_size = np.product([3, 3, 3, 32])
# conv_weights = np.ndarray(
#     shape=[32, 3, 3, 3],
#     dtype='float32',
#     buffer=weights_file.read(weights_size * 4))
# conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])

is_train = True
n_class = 80
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

net = conv2d_unit(net, 512, 1, name='75')
net = conv2d_unit(net, 1024, 3, name='76')
net = conv2d_unit(net, 512, 1, name='77')
net = conv2d_unit(net, 1024, 3, name='78')
net = conv2d_unit(net, 512, 1, name='79')
net = conv2d_unit(net, 1024, 3, name='80')
pred_yolo_1 = conv2d_unit(net, 3 * (5 + n_class), 1, name='81', bn=False)
net = RouteLayer(net, routes=['conv79'], name='83')
net = conv2d_unit(net, 256, 1, name='84')
net = upsample(net, scale=2, name='85')
net = RouteLayer(net, routes=['upsample85', 'res61'], name='86')

net = conv2d_unit(net, 256, 1, name='87')
net = conv2d_unit(net, 512, 3, name='88')
net = conv2d_unit(net, 256, 1, name='89')
net = conv2d_unit(net, 512, 3, name='90')
net = conv2d_unit(net, 256, 1, name='91')
net = conv2d_unit(net, 512, 3, name='92')
pred_yolo_2 = conv2d_unit(net, 3 * (5 + n_class), 1, name='93', bn=False)
net = RouteLayer(net, routes=['conv91'], name='95')
net = conv2d_unit(net, 128, 1, name='96')
net = upsample(net, scale=2, name='97')
net = RouteLayer(net, routes=['upsample97', 'res36'], name='98')

net = conv2d_unit(net, 128, 1, name='99')
net = conv2d_unit(net, 256, 3, name='100')
net = conv2d_unit(net, 128, 1, name='101')
net = conv2d_unit(net, 256, 3, name='102')
net = conv2d_unit(net, 128, 1, name='103')
net = conv2d_unit(net, 256, 3, name='104')
pred_yolo_3 = conv2d_unit(net, 3 * (5 + n_class), 1, name='105', bn=False)

# a = tf.global_variables('bn' + '0')
# with tf.variable_scope('conv0'):
#     b = tf.get_variable(
#         name='b_conv2d', shape=(32,), dtype=tf.float32, initializer=tf.zeros_initializer()
#     )
#     W = tf.get_variable(
#         name='W_conv2d', shape=[3, 3, 3, 32], dtype=tf.float32, initializer=tf.zeros_initializer()
#     )
# with tf.variable_scope('bn0'):
#     gamma = tf.get_variable(
#         'gamma', shape=(32,), initializer=tf.zeros_initializer(), dtype=tf.float32, trainable=is_train,
#     )
#     beta = tf.get_variable(
#         'beta', shape=(32,), initializer=tf.zeros_initializer(), dtype=tf.float32, trainable=is_train
#     )
#     moving_mean = tf.get_variable(
#         'moving_mean', shape=(32,), initializer=tf.zeros_initializer(), dtype=tf.float32, trainable=False
#     )
#     moving_variance = tf.get_variable(
#         'moving_variance', shape=(32,), initializer=tf.constant_initializer(1.), dtype=tf.float32, trainable=False,
#     )

saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in tqdm(range(106)):

        if i in [4, 8, 11, 15, 18, 21, 24, 27, 30, 33, 36, 40, 43, 46, 49, 52, 55, 58, 61, 65, 68, 71, 74]:  # res
            pass
        elif i in [83, 86, 95, 98]:  # route
            pass
        elif i in [82, 94, 106]:  # detction
            pass
        elif i in [85, 97]:  # upsample
            pass
        else:
            tensor_conv_w = tf.global_variables('conv' + str(i))[0]
            tensor_conv_b = tf.global_variables('conv' + str(i))[1]

            in_ch = tensor_conv_w.get_shape().as_list()[-2]
            filter_num = tensor_conv_w.get_shape().as_list()[-1]
            kernel = tensor_conv_w.get_shape().as_list()[0]

            conv_bias = np.ndarray(
                shape=(filter_num,),  # (32,),
                dtype='float32',
                buffer=weights_file.read(filter_num * 4))

            weights_size = np.product([kernel, kernel, in_ch, filter_num])
            conv_weights = np.ndarray(
                shape=[filter_num, in_ch, kernel, kernel],  # [32, 3, 3, 3],
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])

            tf.assign(tensor_conv_w, conv_weights).eval()
            tf.assign(tensor_conv_b, conv_bias).eval()

            if i not in [81, 93, 105]:
                tensor_bn_beta = tf.global_variables('bn' + str(i))[0]
                tensor_bn_gamma = tf.global_variables('bn' + str(i))[1]
                tensor_bn_mean = tf.global_variables('bn' + str(i))[2]
                tensor_bn_variance = tf.global_variables('bn' + str(i))[3]

                bn_weights = np.ndarray(
                    shape=(3, filter_num),  # (3, 32),
                    dtype='float32',
                    buffer=weights_file.read(filter_num * 12))

                tf.assign(tensor_bn_beta, conv_bias).eval()
                tf.assign(tensor_bn_gamma, bn_weights[0]).eval()
                tf.assign(tensor_bn_mean, bn_weights[1]).eval()
                tf.assign(tensor_bn_variance, bn_weights[2]).eval()

    # tf.assign(tf.global_variables('conv0')[0], conv_weights).eval()
    # tf.assign(tf.global_variables('conv0')[1], conv_bias).eval()
    #
    # tf.assign(tf.global_variables('bn0')[0], conv_bias).eval()
    # tf.assign(tf.global_variables('bn0')[1], bn_weights[0]).eval()
    # tf.assign(tf.global_variables('bn0')[2], bn_weights[1]).eval()
    # tf.assign(tf.global_variables('bn0')[3], bn_weights[2]).eval()

    # b = tf.assign(b, conv_bias)
    # W = tf.assign(W, conv_weights)
    # gamma = tf.assign(gamma, bn_weights[0])
    # beta = tf.assign(beta, conv_bias)
    # moving_mean = tf.assign(moving_mean, bn_weights[1])
    # moving_variance = tf.assign(moving_variance, bn_weights[2])
    #
    # sess.run(b)
    # sess.run(W)
    # sess.run(gamma)
    # sess.run(beta)
    # sess.run(moving_mean)
    # sess.run(moving_variance)

    saver.save(sess, checkpoint_dir + "model.ckpt")

# load
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
