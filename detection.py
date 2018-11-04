import tensorflow as tf
import tensorlayer as tl
import cv2
import numpy as np
from tensorlayer.layers import *
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from model import inference
from varible import *


def resize_img(img_path='4.jpg'):
    net_w, net_h = 416, 416
    img = cv2.imread(img_path)
    img = img[:, :, ::-1]  # RGB image
    img_h, img_w = img.shape[0:2]

    scale = min(net_h / img_h, net_w / img_w)
    new_h, new_w = int(img_h * scale), int(img_w * scale)
    offset = ((net_h - new_h) / 2. / net_h, (net_w - new_w) / 2. / net_w)
    scale = (net_h / new_h, net_w / new_w)

    img_sized = cv2.resize(img, (new_w, new_h))  # whc
    img_sized = np.pad(img_sized,
                       (
                           (int((416 - new_h) / 2), 416 - new_h - int((416 - new_h) / 2)),
                           (int((416 - new_w) / 2), 416 - new_w - int((416 - new_w) / 2)),
                           (0, 0)
                       ), mode='constant')
    image_data = np.array(img_sized, dtype='float32')
    return image_data, offset, scale, img_h, img_w


def decode_out(net_out, anchors, offset, scale, img_h, img_w):
    net_w, net_h = 416, 416
    cellbase_x = tf.to_float(tf.reshape(tf.tile(tf.range(52), [52]), (1, 52, 52, 1, 1)))
    cellbase_y = tf.transpose(cellbase_x, (0, 2, 1, 3, 4))
    cellbase_grid = tf.tile(tf.concat([cellbase_x, cellbase_y], -1), [1, 1, 1, 3, 1])

    boxes = list()
    boxes_scores = list()
    for i in range(3):  # 52 26 13
        anchor = anchors[..., 3 * i:3 * (i + 1), :]
        feats = net_out[i]

        grid_w = tf.shape(feats)[2]  # 13
        grid_h = tf.shape(feats)[1]  # 13
        grid_factor = tf.reshape(tf.cast([grid_h, grid_w], tf.float32), [1, 1, 1, 1, 2])
        feats = tf.reshape(feats, [-1, grid_h, grid_w, 3, n_class + 5])

        # Adjust preditions to each spatial grid point and anchor size.
        box_xy = (tf.sigmoid(feats[..., :2]) + cellbase_grid[:, :grid_h, :grid_w, :, :]) / tf.cast(grid_factor,
                                                                                                   'float32')
        box_wh = tf.exp(feats[..., 2:4]) * anchor / tf.cast([net_h, net_w], 'float32')
        box_confidence = tf.sigmoid(feats[..., 4:5])
        box_class_probs = tf.sigmoid(feats[..., 5:])

        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        box_yx = (box_yx - offset) * scale
        box_hw *= scale
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        _boxes = tf.concat([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ], axis=-1)
        # Scale boxes back to original image shape.
        _boxes *= tf.cast((img_h, img_w, img_h, img_w), 'float32')
        boxes = tf.reshape(_boxes, [-1, 4])
        _boxes_scores = box_confidence * box_class_probs
        _boxes_scores = tf.reshape(_boxes_scores, [-1, n_class])

        boxes.append(_boxes)
        boxes_scores.append(_boxes_scores)
        boxes = tf.concat(boxes, axis=0)
        boxes_scores = tf.concat(boxes_scores, axis=0)

    return boxes, boxes_scores


def nms(boxes, boxes_scores):
    mask = boxes_scores >= 0.3
    max_num_boxes = tf.constant(20, dtype='int32')
    boxes_op = []
    scores_op = []
    classes_op = []
    for c in range(n_class):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(boxes_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_num_boxes, iou_threshold=0.5)
        class_boxes = tf.gather(class_boxes, nms_index)
        class_box_scores = tf.gather(class_box_scores, nms_index)
        classes = tf.ones_like(class_box_scores, 'int32') * c
        boxes_op.append(class_boxes)
        scores_op.append(class_box_scores)
        classes_op.append(classes)
    boxes_op = tf.concat(boxes_op, axis=0)
    scores_op = tf.concat(scores_op, axis=0)
    classes_op = tf.concat(classes_op, axis=0)
    return boxes_op, scores_op, classes_op


if __name__ == '__main__':
    checkpoint_dir = './ckpt/'
    ckpt_name = 'ep000-step46000-loss2.157-46000'
    label = Gb_label
    anchors = tf.constant(Gb_anchors, dtype='float', shape=[1, 1, 1, 9, 2])
    n_class = len(label)

    input_pb = tf.placeholder(tf.float32, [None, 416, 416, 3])
    net_out = inference(input_pb, n_class)

    # 读取ckpt里保存的参数
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    # 如果有checkpoint这个文件可以加下面这句话，如果只有一个ckpt文件就不需要这个if
    if tf.train.get_checkpoint_state(checkpoint_dir):  # 确认是否存在
        saver.restore(sess, checkpoint_dir + ckpt_name)
        print("load ok!")
    else:
        print("ckpt文件不存在")

    image_data, offset, scale, img_h, img_w = resize_img(img_path='4.jpg')
    boxes, boxes_scores = decode_out(net_out, anchors, offset, scale, img_h, img_w)
    boxes_op, scores_op, classes_op = nms(boxes, boxes_scores)
    b, s, c = sess.run([boxes_op, scores_op, classes_op], feed_dict={input_pb: image_data})

    plt.cla()
    plt.imshow(img)
    for i, obj in enumerate(b):
        x1 = obj[1]
        x2 = obj[3]
        y1 = obj[0]
        y2 = obj[2]

        # TODO: change the color of text
        plt.text(x1, y1 - 10, round(s[i], 2))
        plt.text(x2 - 30, y1 - 10, label[c[i]])
        plt.hlines(y1, x1, x2, colors='red')
        plt.hlines(y2, x1, x2, colors='red')
        plt.vlines(x1, y1, y2, colors='red')
        plt.vlines(x2, y1, y2, colors='red')
    plt.show()
    exit()
#
# if not os.path.exists('out'):
#     os.mkdir('out')
# file_name = input('Input image filedir:')
# img_path = os.listdir(file_name)
# for path in tqdm(img_path):
#     abs_path = file_name + path
#     img = cv2.imread(abs_path)
#     # while True:
#     #     file_name = input('Input image filename:')
#     #     img = cv2.imread(file_name)
#     img = img[:, :, ::-1]  # RGB image
#     img_shape = img.shape[0:2][::-1]
#
#     _scale = min(416 / img_shape[0], 416 / img_shape[1])
#     _new_shape = (int(img_shape[0] * _scale), int(img_shape[1] * _scale))
#     im_sized = cv2.resize(img, _new_shape)
#     im_sized = np.pad(im_sized,
#                       (
#                           (int((416 - _new_shape[1]) / 2), 416 - _new_shape[1] - int((416 - _new_shape[1]) / 2)),
#                           (int((416 - _new_shape[0]) / 2), 416 - _new_shape[0] - int((416 - _new_shape[0]) / 2)),
#                           (0, 0)
#                       ),
#                       mode='constant')
#     image_data = np.array(im_sized, dtype='float32')
#     image_data /= 255.
#     image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
#
#     input_shape = tf.cast(tf.shape(net_out[2])[1:3] * 32, dtype='float32')[::-1]  # hw
#     image_shape = tf.cast(img_shape, dtype='float32')[::-1]  # hw
#     new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
#     offset = (input_shape - new_shape) / 2. / input_shape
#     scale = input_shape / new_shape
#
#     # with tf.Session() as sess:
#     #     a = sess.run(scale)
#
#     boxes = list()
#     box_scores = list()
#
#     cellbase_x = tf.to_float(tf.reshape(tf.tile(tf.range(52), [52]), (1, 52, 52, 1, 1)))
#     cellbase_y = tf.transpose(cellbase_x, (0, 2, 1, 3, 4))
#     cellbase_grid = tf.tile(tf.concat([cellbase_x, cellbase_y], -1), [1, 1, 1, 3, 1])
#     # classes = list()
#     for i in range(3):  # 52 26 13
#         anchor = anchors[..., 3 * i:3 * (i + 1), :]
#         # feats = model.output[i]
#         feats = net_out[i]
#
#         grid_w = tf.shape(feats)[1]  # 13
#         grid_h = tf.shape(feats)[2]  # 13
#         grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 2])
#
#         feats = tf.reshape(feats, [-1, grid_w, grid_h, 3, n_class + 5])
#
#         # Adjust preditions to each spatial grid point and anchor size.
#         box_xy = (tf.sigmoid(feats[..., :2]) + cellbase_grid[:, :grid_w, :grid_h, :, :]) / tf.cast(grid_factor[::-1],
#                                                                                                    'float32')
#         box_wh = tf.exp(feats[..., 2:4]) * anchor / tf.cast(input_shape[::-1], 'float32')
#         box_confidence = tf.sigmoid(feats[..., 4:5])
#         box_class_probs = tf.sigmoid(feats[..., 5:])
#
#         box_yx = box_xy[..., ::-1]
#         box_hw = box_wh[..., ::-1]
#         box_yx = (box_yx - offset) * scale
#         box_hw *= scale
#         box_mins = box_yx - (box_hw / 2.)
#         box_maxes = box_yx + (box_hw / 2.)
#         _boxes = tf.concat([
#             box_mins[..., 0:1],  # y_min
#             box_mins[..., 1:2],  # x_min
#             box_maxes[..., 0:1],  # y_max
#             box_maxes[..., 1:2]  # x_max
#         ], axis=-1)
#
#         # Scale boxes back to original image shape.
#         _boxes *= tf.concat([tf.cast(image_shape, 'float32'), tf.cast(image_shape, 'float32')], axis=-1)
#         _boxes = tf.reshape(_boxes, [-1, 4])
#
#         _box_scores = box_confidence * box_class_probs
#         _box_scores = tf.reshape(_box_scores, [-1, n_class])
#         boxes.append(_boxes)
#         box_scores.append(_box_scores)
#     boxes = tf.concat(boxes, axis=0)
#     box_scores = tf.concat(box_scores, axis=0)
#
#     mask = box_scores >= 0.3
#     max_num_boxes = tf.constant(20, dtype='int32')
#
#     boxes_ = []
#     scores_ = []
#     classes_ = []
#     for c in range(n_class):
#         class_boxes = tf.boolean_mask(boxes, mask[:, c])
#         class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
#         nms_index = tf.image.non_max_suppression(
#             class_boxes, class_box_scores, max_num_boxes, iou_threshold=0.5)
#         class_boxes = tf.gather(class_boxes, nms_index)
#         class_box_scores = tf.gather(class_box_scores, nms_index)
#         classes = tf.ones_like(class_box_scores, 'int32') * c
#         boxes_.append(class_boxes)
#         scores_.append(class_box_scores)
#         classes_.append(classes)
#     boxes_ = tf.concat(boxes_, axis=0)
#     scores_ = tf.concat(scores_, axis=0)
#     classes_ = tf.concat(classes_, axis=0)
#
#     b, s, c = sess.run([boxes_, scores_, classes_], feed_dict={input_pb: image_data})
#
#     # plt.cla()
#     # plt.imshow(img)
#     # for i, obj in enumerate(b):
#     #     x1 = obj[1]
#     #     x2 = obj[3]
#     #     y1 = obj[0]
#     #     y2 = obj[2]
#     #
#     #     # TODO: change the color of text
#     #     plt.text(x1, y1 - 10, round(s[i], 2))
#     #     plt.text(x2 - 30, y1 - 10, label[c[i]])
#     #     plt.hlines(y1, x1, x2, colors='red')
#     #     plt.hlines(y2, x1, x2, colors='red')
#     #     plt.vlines(x1, y1, y2, colors='red')
#     #     plt.vlines(x2, y1, y2, colors='red')
#     # plt.show()
#     img = img[:, :, ::-1]
#     file = open("./out/" + path.rstrip('.jpg') + '.txt', 'w')
#     for i, obj in enumerate(b):
#         cv2.rectangle(img, (obj[1], obj[0]), (obj[3], obj[2]), (0, 0, 255), 3)
#         cv2.putText(img, str(round(s[i], 2)), (int(obj[1]), int(obj[0]) - 10), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255),
#                     3)
#         cv2.putText(img, str(label[c[i]]), (int(obj[3]) - 100, int(obj[0]) - 10), cv2.FONT_HERSHEY_COMPLEX, 2,
#                     (0, 0, 255), 3)
#
#         file.write('{0} {1} '.format(label[c[i]], s[i]))
#         file.write('{0} {1} {2} {3}'.format(obj[1], obj[0], obj[3], obj[2]))
#         file.write('\n')
#     file.close()
#     cv2.imwrite("./out/" + path, img)
#     # cv2.imwrite("1.jpg", img)
