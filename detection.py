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


def resize_img(img_path):
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
    image_data = np.array(img_sized, dtype='float32') / 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return img, image_data, offset, scale, img_h, img_w


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
        _boxes = tf.reshape(_boxes, [-1, 4])
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


def show_result(img_original, b, s, c, out_dir, file_path, out_mode=0):
    if out_mode == 0:
        plt.cla()
        plt.imshow(img_original)
        for i, obj in enumerate(b):
            x1 = obj[1]
            x2 = obj[3]
            y1 = obj[0]
            y2 = obj[2]
            # TODO: change the color of text
            # plt.text(x1, y1 - 10, round(s[i], 2))
            plt.text(x2 - 30, y1 - 10, label[c[i]])
            plt.hlines(y1, x1, x2, colors='red')
            plt.hlines(y2, x1, x2, colors='red')
            plt.vlines(x1, y1, y2, colors='red')
            plt.vlines(x2, y1, y2, colors='red')
        plt.show()
    elif out_mode == 1:
        file = open(out_dir + file_path.split('/')[-1][0:-4] + '.txt', 'w')
        img_original = img_original.copy()
        for i, obj in enumerate(b):
            cv2.rectangle(img_original, (obj[1], obj[0]), (obj[3], obj[2]), (0, 0, 255), 1)
            # cv2.putText(img_original, str(round(s[i], 2)), (int(obj[1]), int(obj[0]) - 10), cv2.FONT_HERSHEY_COMPLEX, 2,
            #             (0, 0, 255), 1)
            cv2.putText(img_original, str(label[c[i]]), (int(obj[3]) - 100, int(obj[0]) - 10), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 0, 255), 1)

            file.write('{0} {1} '.format(label[c[i]], s[i]))
            file.write('{0} {1} {2} {3}'.format(obj[1], obj[0], obj[3], obj[2]))
            file.write('\n')
        file.close()
        cv2.imwrite(out_dir + file_path.split('/')[-1], img_original)


def remove_outbox(b, s, c, img_h, img_w):
    b = list(b)
    s = list(s)
    c = list(c)
    k = 0
    while (k < len(b)):
        if b[k][0] < 0 or b[k][1] < 0 or b[k][2] > img_h or b[k][3] > img_w:
            b.pop(k)
            s.pop(k)
            c.pop(k)
            k -= 1
        k += 1
    return b, s, c


def iou(pre_boxes, true_boxes):
    pred_mins = pre_boxes[..., :2]
    pred_maxes = pre_boxes[..., 2:4]
    true_mins = true_boxes[..., :2]
    true_maxes = true_boxes[..., 2:4]

    #     intersect_mins = tf.maximum(pred_mins, true_mins)
    #     intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_mins = np.maximum(pred_mins, true_mins)
    intersect_maxes = np.minimum(pred_maxes, true_maxes)
    #     print(intersect_mins.shape)
    #     print(intersect_maxes.shape)

    #     intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)

    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = (true_maxes[..., 0] - true_mins[..., 0]) * (true_maxes[..., 1] - true_mins[..., 1])
    pred_areas = (pred_maxes[..., 0] - pred_mins[..., 0]) * (pred_maxes[..., 1] - pred_mins[..., 1])

    union_areas = pred_areas + true_areas - intersect_areas
    iou_score = np.divide(intersect_areas, union_areas)

    return iou_score


def remove_boxes_among_classes(boxes, score, box_class, threshold=0.5):
    boxes = np.array(boxes)
    score = np.array(score)
    box_class = np.array(box_class)

    remove_index = np.zeros(boxes.shape[0])
    iou_list = []
    for i in range(boxes.shape[0]):
        for j in range(i + 1, boxes.shape[0]):
            temp = iou(boxes[i], boxes[j])
            iou_list.append(temp)
            if (temp > threshold):
                if (score[i] > score[j]):
                    remove_index[j] = 1
                else:
                    remove_index[i] = 1

    for i in reversed(range(len(remove_index))):
        if (remove_index[i] == 1):
            boxes = np.delete(boxes, i, axis=0)
            score = np.delete(score, i, axis=0)
            box_class = np.delete(box_class, i, axis=0)

    return boxes, score, box_class


if __name__ == '__main__':
    checkpoint_dir = './ckpt/'
    ckpt_name = 'ep3138-step56500-loss22.640'
    label = Gb_label
    anchors = tf.constant(Gb_anchors, dtype='float', shape=[1, 1, 1, 9, 2])
    n_class = len(label)

    input_pb = tf.placeholder(tf.float32, [None, 416, 416, 3])
    net_out = inference(input_pb, n_class)

    # 读取ckpt里保存的参数
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    try:
        saver.restore(sess, checkpoint_dir + ckpt_name)
        print("load ok!")
    except:
        print("ckpt文件不存在")
        raise

    detection_mode = 0  # 0一张张, 1全部
    out_mode = 0  # 0显示，1存储
    out_dir = './out/'
    # detect img one by one
    if detection_mode == 0:
        while True:
            file_path = input('Input file_path:')
            img, image_data, offset, scale, img_h, img_w = resize_img(file_path)
            boxes, boxes_scores = decode_out(net_out, anchors, offset, scale, img_h, img_w)
            boxes_op, scores_op, classes_op = nms(boxes, boxes_scores)
            b, s, c = sess.run([boxes_op, scores_op, classes_op], feed_dict={input_pb: image_data})
            b, s, c = remove_outbox(b, s, c, img_h, img_w)
            b, s, c = remove_boxes_among_classes(b, s, c, threshold=0.2)
            show_result(img, b, s, c, out_dir=out_dir, file_path=file_path, out_mode=out_mode)
    # detect all files
    elif detection_mode == 1:
        imgs_paths = list()
        txt_path = Gb_data_dir + 'val.txt'
        with open(txt_path) as fh:
            for line in fh:
                imgs_paths.append(Gb_data_dir + 'images/' + line.strip())

        for img_path in imgs_paths:
            img, image_data, offset, scale, img_h, img_w = resize_img(img_path)
            boxes, boxes_scores = decode_out(net_out, anchors, offset, scale, img_h, img_w)
            # for i in range(Gb_batch_size):
            boxes_op, scores_op, classes_op = nms(boxes, boxes_scores)
            b, s, c = sess.run([boxes_op, scores_op, classes_op], feed_dict={input_pb: image_data})
            b, s, c = remove_outbox(b, s, c, img_h, img_w)
            b, s, c = remove_boxes_among_classes(b, s, c, threshold=0.2)
            show_result(img, b, s, c, out_dir=out_dir, file_path=img_path, out_mode=out_mode)
    exit()
