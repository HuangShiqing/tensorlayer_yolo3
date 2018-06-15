from net import *
# import tensorflow as tf

logger = log.getLogger('tensorlayer')
logger.setLevel(level=log.ERROR)

checkpoint_dir = 'D:/DeepLearning/code/tensorlayer_yolo3/ckpt/'

# load
input_pb = tf.placeholder(tf.float32, [None, 416, 416, 3])
net = InputLayer(input_pb, name='input')
net = conv2d_unit(net, 32, 3, name='0')

net = conv2d_unit(net, 64, 3, strides=2, name='1')
net = stack_residual_block(net, 32, 1, name='2')

net = conv2d_unit(net, 128, 3, strides=2, name='5')
net = stack_residual_block(net, 64, 2, name='6')

net = conv2d_unit(net, 256, 3, strides=2, name='12')


# 读取ckpt里保存的参数
sess = tf.InteractiveSession()
saver = tf.train.Saver()
# 如果有checkpoint这个文件可以加下面这句话，如果只有一个ckpt文件就不需要这个if
if tf.train.get_checkpoint_state(checkpoint_dir):  # 确认是否存在
    saver.restore(sess, checkpoint_dir + "model.ckpt")
    print("load ok!")
else:
    print("ckpt文件不存在")

a = tf.global_variables()
conv0_w = tf.global_variables()[0].eval()
bn0_beta = tf.global_variables()[2].eval()
bn0_mean = tf.global_variables()[4].eval()
conv1_w = tf.global_variables()[6].eval()
bn1_beta = tf.global_variables()[8].eval()
bn1_mean = tf.global_variables()[10].eval()
bn5_var = tf.global_variables()[-1].eval()

# b = tf.global_variables()[-1].eval()
exit()