import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from net import ResLayer, RouteLayer, upsample, conv2d_unit, residual_block, stack_residual_block

import cv2
import numpy as np

import logging as log

from PIL import Image

logger = log.getLogger('tensorlayer')
logger.setLevel(level=log.ERROR)

checkpoint_dir = './ckpt/'
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
pred_yolo_1 = conv2d_unit(net, 3 * (5 + n_class), 1, name='81', bn=False).outputs
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
pred_yolo_2 = conv2d_unit(net, 3 * (5 + n_class), 1, name='93', bn=False).outputs
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
pred_yolo_3 = conv2d_unit(net, 3 * (5 + n_class), 1, name='105', bn=False).outputs

# 读取ckpt里保存的参数
sess = tf.InteractiveSession()
saver = tf.train.Saver()
# 如果有checkpoint这个文件可以加下面这句话，如果只有一个ckpt文件就不需要这个if
if tf.train.get_checkpoint_state(checkpoint_dir):  # 确认是否存在
    saver.restore(sess, checkpoint_dir + "model.ckpt")
    print("load ok!")
else:
    print("ckpt文件不存在")

# image = Image.open('dog.jpg')
# image = image.resize((416, 416), Image.BICUBIC)
# image_data = np.array(image, dtype='float32')
# image_data = image_data / 255.
# image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
img = cv2.imread('dog.jpg')
img = img[:, :, ::-1]  # RGB image
im_sized = cv2.resize(img, (416, 416))
image_data = np.array(im_sized, dtype='float32')
image_data /= 255.
image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

pred = [pred_yolo_3, pred_yolo_2, pred_yolo_1]

# [pred[0], pred[1], pred[2]] = sess.run([pred_yolo_1, pred_yolo_2, pred_yolo_3], feed_dict={input_pb: image_data})

cellbase_x = tf.to_float(tf.reshape(tf.tile(tf.range(52), [52]), (1, 52, 52, 1, 1)))
cellbase_y = tf.transpose(cellbase_x, (0, 2, 1, 3, 4))
cellbase_grid = tf.tile(tf.concat([cellbase_x, cellbase_y], -1), [1, 1, 1, 3, 1])

img_w = 416
img_h = 416
img_factor = tf.reshape(tf.cast([img_w, img_h], tf.float32), [1, 1, 1, 1, 2])

anchors = tf.constant([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                      dtype='float', shape=[1, 1, 1, 9, 2])
boxes = list()
scores = list()
classes = list()
for i in range(3):
    anchor = anchors[..., 3 * i:3 * (i + 1), :]

    grid_w = tf.shape(pred[i])[1]  # 13
    grid_h = tf.shape(pred[i])[2]  # 13
    grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 2])

    net_out_reshape = tf.reshape(pred[i], [-1, grid_w, grid_h, 3, (4 + 1 + n_class)])
    adjusted_out_xy = (cellbase_grid[:, :grid_w, :grid_h, :, :] + tf.sigmoid(net_out_reshape[..., :2])) / grid_factor
    adjusted_out_wh = tf.exp(net_out_reshape[..., 2:4]) * anchor / img_factor
    adjusted_out_c = tf.expand_dims(tf.sigmoid(net_out_reshape[..., 4]), axis=-1)
    adjusted_out_class = tf.sigmoid(net_out_reshape[..., 5:])

    # offset = ([input_shape - [416, 416]) / 2. / input_shape
    # scale = input_shape / [416, 416]
    # adjusted_out_xy = (adjusted_out_xy - offset) * scale
    # adjusted_out_wh *= scale

    box_mins = adjusted_out_xy - (adjusted_out_wh / 2.)
    box_maxes = adjusted_out_xy + (adjusted_out_wh / 2.)
    box = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)

    # Scale boxes back to original image shape.
    # box *= tf.concat([input_shape, input_shape])
    box *= [416, 416, 416, 416]
    box = tf.reshape(box, [-1, 4])

    box_scores = adjusted_out_c * adjusted_out_class
    max_scores = tf.reduce_max(box_scores, axis=-1)
    max_ind = tf.argmax(box_scores, axis=-1)

    max_scores = tf.reshape(max_scores, [-1, ])
    max_ind = tf.reshape(max_ind, [-1, ])

    boxes.append(box)
    scores.append(max_scores)
    classes.append(max_ind)

[b, s] = sess.run([boxes, scores], feed_dict={input_pb: image_data})
# im_sized = im_sized[:, :, ::-1]  # RGB image
for m in range(3):
    for k in range(len(s[m])):
        if s[m][k] > 0.5:
            cv2.rectangle(im_sized, (b[m][1] * 416, b[m][0] * 416), (b[m][3] * 416, b[m][2] * 416), (0, 255, 0), 1)
cv2.imwrite("./0.jpg", img)

sess.close()
# box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
#         anchors, num_classes, input_shape)
#     boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
#     boxes = K.reshape(boxes, [-1, 4])
#     box_scores = box_confidence * box_class_probs
#     box_scores = K.reshape(box_scores, [-1, num_classes])
