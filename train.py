import numpy as np
import tensorflow as tf


def iou():


B = 3
C = 2

pred_yolo_1 = tf.placeholder(tf.float32, [None, 13, 13, 3 * (5 + C)])
y_true_1 = tf.placeholder(tf.float32, [None, 13, 13, 3, 5 + C])
origin_boxes = tf.placeholder(tf.float32, [None, 1, 1, 1, 8, 4])
# pred_yolo_2 = tf.placeholder(tf.float32, [None, 26, 26, 255])
# pred_yolo_3 = tf.placeholder(tf.float32, [None, 52, 52, 255])


grid_h = tf.shape(y_true_1)[1]
grid_w = tf.shape(y_true_1)[2]

net_out_reshape = tf.reshape(pred_yolo_1, [-1, 13, 13, 3, (4 + 1 + C)])

pred_box = net_out_reshape[..., 0:4]
