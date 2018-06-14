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
    iou_score = tf.truediv(intersect_areas, union_areas)

    return iou_score


B = 3
C = 2
batch_size = 16
input_size = [416, 416]
ignore_thresh = 0.5
obj_scale = 5
noobj_scale = 1
# xywh_scale = 1
class_scale = 1
# anchors = tf.constant([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326], dtype='float',
#                       shape=[1, 1, 1, 9, 2])
anchors = tf.constant([208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208, 208],
                      dtype='float', shape=[1, 1, 1, 9, 2])

y_pred = list()
y_true = list()

y_pred.append(tf.tile(tf.reshape([-1.098, -1.098, 1, 1, 0.8, 0.2, 0.3], shape=[1, 1, 1, 1, 7]), [16, 13, 13, 3, 1]))
# y_pred.append(tf.placeholder(tf.float32, [None, 13, 13, 3 * (5 + C)]))
y_pred.append(tf.placeholder(tf.float32, [None, 26, 26, 3 * (5 + C)]))
y_pred.append(tf.placeholder(tf.float32, [None, 52, 52, 3 * (5 + C)]))

y_true.append(tf.placeholder(tf.float32, [None, 13, 13, 3, 5 + C]))
y_true.append(tf.placeholder(tf.float32, [None, 26, 26, 3, 5 + C]))
y_true.append(tf.placeholder(tf.float32, [None, 52, 52, 3, 5 + C]))
# origin_boxes = tf.placeholder(tf.float32, [None, 1, 1, 1, 8, 4])
# origin_boxes = tf.tile(tf.reshape([200.0, 200.0, 208.0, 208.0], shape=[1, 1, 1, 1, 1, 4]), [16, 1, 1, 1, 8, 1])
# pred_yolo_2 = tf.placeholder(tf.float32, [None, 26, 26, 255])
# pred_yolo_3 = tf.placeholder(tf.float32, [None, 52, 52, 255])

cellbase_x = tf.to_float(tf.reshape(tf.tile(tf.range(52), [52]), (1, 52, 52, 1, 1)))
cellbase_y = tf.transpose(cellbase_x, (0, 2, 1, 3, 4))
cellbase_grid = tf.tile(tf.concat([cellbase_x, cellbase_y], -1), [batch_size, 1, 1, 3, 1])

img_w = 416
img_h = 416
img_factor = tf.reshape(tf.cast([img_w, img_h], tf.float32), [1, 1, 1, 1, 2])

loss = 0
for i in range(3):
    # TODO:adjust the order of the anchor
    anchor = anchors[..., 3 * i:3 * (i + 1), :]
    object_mask = tf.expand_dims(y_true[i][..., 4], 4)

    grid_w = tf.shape(y_pred[i])[1]  # 13
    grid_h = tf.shape(y_pred[i])[2]  # 13
    grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 2])

    net_out_reshape = tf.reshape(y_pred[i], [-1, grid_w, grid_h, 3, (4 + 1 + C)])
    adjusted_out_xy = (cellbase_grid[:, :grid_w, :grid_h, :, :] + tf.sigmoid(net_out_reshape[..., :2])) / grid_factor
    adjusted_out_wh = tf.exp(net_out_reshape[..., 2:4]) * anchor / img_factor
    adjusted_out_c = tf.expand_dims(tf.sigmoid(net_out_reshape[..., 4]), axis=-1)
    adjusted_out_class = tf.sigmoid(net_out_reshape[..., 5:])
    adjusted_net_out = tf.concat([adjusted_out_xy, adjusted_out_wh, adjusted_out_c, adjusted_out_class], axis=-1)
    pred_boxes = tf.expand_dims(adjusted_net_out[..., 0:4], 4)

    # adjusted_origin_xy = origin_boxes[..., :2] / img_factor
    # adjusted_origin_wh = origin_boxes[..., 2:4] / img_factor
    # adjusted_origin_boxes = tf.concat([adjusted_origin_xy, adjusted_origin_wh], axis=-1)
    origin_boxes = tf.boolean_mask(y_true[i][..., :4], y_true[i][..., 4])
    origin_boxes = tf.tile(tf.reshape(origin_boxes, shape=[1, 1, 1, 1, -1, 4]), [batch_size, 1, 1, 1, 1, 1])
    iou_scores = iou(pred_boxes, origin_boxes)
    best_ious = tf.reduce_max(iou_scores, axis=4)
    conf_delta = adjusted_out_c
    conf_delta *= tf.expand_dims(tf.to_float(best_ious < ignore_thresh), 4)

    adjusted_true_xy = y_true[i][..., :2] * grid_factor - cellbase_grid[:, :grid_w, :grid_h, :, :]
    adjusted_true_wh = tf.log(y_true[i][..., 2:4] / anchor * img_factor)
    # TODO:avoid log(0) = -inf
    adjusted_true_c = tf.expand_dims(y_true[i][..., 4], 4)
    # adjusted_true_class = tf.argmax(y_true[i][..., 5:], -1)
    adjusted_true_class = y_true[i][..., 5:]
    # adjusted_true_boxes = tf.concat([adjusted_true_xy, adjusted_true_wh], axis=-1)
    # pred_boxes = adjusted_net_out[..., 0:4]

    xywh_scale = 2 - y_true[i][..., 2:3] * y_true[i][..., 3:4]
    # iou_scores = iou(pred_boxes, adjusted_true_boxes)
    # iou_scores = object_mask * tf.expand_dims(iou_scores, 4)

    loss_xy = tf.reduce_sum(
        object_mask * xywh_scale * tl.cost.binary_cross_entropy(adjusted_out_xy, adjusted_true_xy)) / batch_size
    loss_wh = tf.reduce_sum(
        object_mask * xywh_scale * 0.5 * tf.square(adjusted_out_wh - adjusted_true_wh)) / batch_size
    loss_c = tf.reduce_sum(object_mask * tl.cost.binary_cross_entropy(adjusted_out_c, adjusted_true_c) * obj_scale + (
            1 - object_mask) * conf_delta * noobj_scale) / batch_size
    loss_class = tf.reduce_sum(
        object_mask * tl.cost.binary_cross_entropy(adjusted_out_class, adjusted_true_class) * class_scale) / batch_size
    loss += loss_xy + loss_wh + loss_c + loss_class
