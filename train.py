import numpy as np
import tensorflow as tf
import tensorlayer as tl


def iou(pre_boxes, true_boxes):
    pred_xy = pre_boxes[..., :2]
    pred_wh = pre_boxes[..., 2:4]
    true_xy = true_boxes[..., :2]
    true_wh = true_boxes[..., 2:4]

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)

    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    return iou_scores


B = 3
C = 2
batch_size = 16
input_size = [416, 416]
ignore_thresh = 0.5
obj_scale = 5
noobj_scale = 1
xywh_scale = 1
class_scale = 1
# anchors = tf.constant([228, 184, 285, 359, 341, 260], dtype='float', shape=[1, 1, 1, 3, 2])
anchors = tf.constant([208, 208, 208, 208, 208, 208], dtype='float', shape=[1, 1, 1, 3, 2])

# pred_yolo_1 = tf.placeholder(tf.float32, [None, 13, 13, 3 * (5 + C)])
pred_yolo_1 = tf.tile(tf.reshape([-1.098, -1.098, 1, 1, 0.8, 0.2, 0.3], shape=[1, 1, 1, 1, 7]), [16, 13, 13, 3, 1])
y_true_1 = tf.placeholder(tf.float32, [None, 13, 13, 3, 5 + C])
# y_true_1 = tf.tile(tf.reshape([-1.098, -1.098, 1, 1, 0.8, 0.2, 0.3], shape=[1, 1, 1, 1, 7]), [16, 13, 13, 3, 1])
# origin_boxes = tf.placeholder(tf.float32, [None, 1, 1, 1, 8, 4])
origin_boxes = tf.tile(tf.reshape([200.0, 200.0, 208.0, 208.0], shape=[1, 1, 1, 1, 1, 4]), [16, 1, 1, 1, 8, 1])
# pred_yolo_2 = tf.placeholder(tf.float32, [None, 26, 26, 255])
# pred_yolo_3 = tf.placeholder(tf.float32, [None, 52, 52, 255])

object_mask = tf.expand_dims(y_true_1[..., 4], 4)

cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(13), [13]), (1, 13, 13, 1, 1)))
cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [batch_size, 1, 1, 3, 1])

grid_h = tf.shape(pred_yolo_1)[1]
grid_w = tf.shape(pred_yolo_1)[2]
grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 2])
net_h = 416
net_w = 416
net_factor = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1, 1, 1, 1, 2])

net_out_reshape = tf.reshape(pred_yolo_1, [-1, 13, 13, 3, (4 + 1 + C)])
adjusted_out_xy = (cell_grid + tf.sigmoid(net_out_reshape[..., :2])) / grid_factor
adjusted_out_wh = net_out_reshape[..., 2:4] * anchors / net_factor
adjusted_out_c = tf.expand_dims(tf.sigmoid(net_out_reshape[..., 4]), axis=-1)
adjusted_out_class = net_out_reshape[..., 5:]
adjusted_net_out = tf.concat([adjusted_out_xy, adjusted_out_wh, adjusted_out_c, adjusted_out_class], axis=-1)
pred_boxes = tf.expand_dims(adjusted_net_out[..., 0:4], 4)

adjusted_origin_xy = origin_boxes[..., :2] / net_factor
adjusted_origin_wh = origin_boxes[..., 2:4] / net_factor
adjusted_origin_boxes = tf.concat([adjusted_origin_xy, adjusted_origin_wh], axis=-1)
iou_scores = iou(pred_boxes, adjusted_origin_boxes)
best_ious = tf.reduce_max(iou_scores, axis=4)
conf_delta = adjusted_out_c
conf_delta *= tf.expand_dims(tf.to_float(best_ious < ignore_thresh), 4)

adjusted_label_xy = y_true_1[..., :2] / grid_factor
adjusted_label_wh = tf.exp(y_true_1[..., 2:4]) * anchors / net_factor
adjusted_label_c = tf.expand_dims(y_true_1[..., 4], 4)
adjusted_label_class = tf.argmax(y_true_1[..., 5:], -1)
adjusted_label_boxes = tf.concat([adjusted_label_xy, adjusted_label_wh], axis=-1)
pred_boxes = adjusted_net_out[..., 0:4]

iou_scores = iou(pred_boxes, adjusted_label_boxes)
iou_scores = object_mask * tf.expand_dims(iou_scores, 4)

loss_xy = tf.reduce_sum(
    object_mask * tl.cost.binary_cross_entropy(adjusted_out_xy, adjusted_label_xy) * xywh_scale) / batch_size
loss_wh = tf.reduce_sum(
    object_mask * tl.cost.binary_cross_entropy(adjusted_out_wh, adjusted_label_wh) * xywh_scale) / batch_size
loss_c = tf.reduce_sum(object_mask * tl.cost.binary_cross_entropy(adjusted_out_c, adjusted_label_c) * obj_scale + (
        1 - object_mask) * conf_delta * noobj_scale) / batch_size
loss_class = tf.reduce_sum(
    object_mask * tl.cost.binary_cross_entropy(adjusted_out_class, adjusted_label_class) * class_scale) / batch_size
