import os
import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt  # dealing with plots
from tqdm import tqdm

from varible import *


def read_xml(txt_name, pick):
    """
        读取目录下所有的xml文件，获得pick标注框的信息

        Parameters
        ----------
        txt_name : str
        pick : list

        Returns
        -------
        chunks : list

        Examples
        --------
        txt_name = 'train.txt'
        pick = ['die knot','live knot']
        chunks = [['1065.jpg', [2048, 1536, [{'name': 'die knot', 'ymax': 819, 'xmin': 1026, 'xmax': 1478, 'ymin': 482},
                                            {'name': 'die knot', 'ymax': 714, 'xmin': 796, 'xmax': 986, 'ymin': 387}]]],
                  ['10.jpg', [2048, 1536, [{'ymax': 1074, 'name': 'die knot', 'xmax': 982, 'xmin': 478, 'ymin': 535}]]]]
    """
    print('Parsing for {}'.format(pick))
    chunks = list()
    txt_path = Gb_data_dir + txt_name
    annotations = []
    with open(txt_path) as fh:
        for line in fh:
            annotations.append(Gb_data_dir + 'labels/' + line[:-4] + 'xml')
    not_in_pick = dict()

    for file in tqdm(annotations):
        # actual parsing
        in_file = open(file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        jpg = str(root.find('filename').text)
        jpg = jpg + '.jpg' if 'jpg' not in jpg else ''
        imsize = root.find('size')
        w = int(imsize.find('width').text)
        h = int(imsize.find('height').text)
        all = list()

        for obj in root.iter('object'):
            # current = list()
            current = dict()
            name = obj.find('name').text
            if name not in pick:
                if name not in not_in_pick:
                    not_in_pick[name] = 1
                else:
                    not_in_pick[name] += 1
                continue

            xmlbox = obj.find('bndbox')
            xn = int(float(xmlbox.find('xmin').text))
            xx = int(float(xmlbox.find('xmax').text))
            yn = int(float(xmlbox.find('ymin').text))
            yx = int(float(xmlbox.find('ymax').text))
            # current = [name, xn, yn, xx, yx]
            current['name'] = name
            current['xmin'] = xn
            current['xmax'] = xx
            current['ymin'] = yn
            current['ymax'] = yx
            all += [current]

        add = [[jpg, [w, h, all]]]
        if len(all) is not 0:  # skip the image which not include any 'pick'
            chunks += add
        in_file.close()

    # gather all stats
    stat = dict()
    for chunk in chunks:
        all = chunk[1][2]
        for current in all:
            if current['name'] in pick:
                if current['name'] in stat:
                    stat[current['name']] += 1
                else:
                    stat[current['name']] = 1

    print('\nPick:')
    for i in stat: print('    {}: {}'.format(i, stat[i]))
    print('Not in pick:')
    for j in not_in_pick: print('    {}: {}'.format(j, not_in_pick[j]))
    print('Boxes size: {}'.format(len(chunks)))

    return chunks


def random_flip(image, flip):
    """
                    随机左右翻转图片

                    Parameters
                    ----------
                    image : 改变前的原图 RGB模式 ndarray [h,w,3] dtype=uint8
                    flip : int 0或者1

                    Returns
                    -------
                    image : 改变后的图 RGB模式 ndarray [h,w,3] dtype=uint8

                    Examples
                    --------

                """
    if flip == 1:
        return cv2.flip(image, 1)
    return image


def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5):
    """
                随机改变图像色调、饱和度、明亮度

                Parameters
                ----------
                image : 改变前的原图 RGB模式 ndarray [h,w,3] dtype=uint8
                hue : float
                saturation : float
                exposure : float

                Returns
                -------
                改变后的图 RGB模式 ndarray [h,w,3] dtype=uint8

                Examples
                --------

            """

    def _rand_scale(scale):
        scale = np.random.uniform(1, scale)
        return scale if (np.random.randint(2) == 0) else 1. / scale

    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = _rand_scale(saturation)
    dexp = _rand_scale(exposure)
    # convert RGB space to HSV space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')
    # change satuation and exposure
    image[:, :, 1] *= dsat
    image[:, :, 2] *= dexp
    # change hue
    image[:, :, 0] += dhue
    image[:, :, 0] -= (image[:, :, 0] > 180) * 180
    image[:, :, 0] += (image[:, :, 0] < 0) * 180
    # avoid overflow when astype('uint8')
    image[...] = np.clip(image[...], 0, 255)
    # convert back to RGB from HSV
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)


def adjust_wh_ratio(image_h, image_w, net_h, net_w):
    """
            小幅随机改变图像长宽比，改变幅度取决于Gb_jitter

            Parameters
            ----------
            image_h : int 原图的h
            image_w : int 原图的w
            net_h : int  输入网络的图像的h，常用的是416
            net_w : int 输入网络的图像的，常用的是416

            Returns
            -------
            new_h : int 缩放后的h
            new_w : int 缩放后的w
            dx : int
            dy : int

            Examples
            --------
            image_h = 2048
            image_w = 1536
            net_h = 416
            net_w = 416

            new_h = 451
            new_w = 276
            dx = 19
            dy = -23

        """
    jitter = Gb_jitter

    # determine the amount of scaling and cropping
    dw = jitter * image_w
    dh = jitter * image_h

    new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh))
    scale = np.random.uniform(1, 1.5)

    if new_ar < 1:
        new_h = int(scale * net_h)
        new_w = int(net_h * new_ar)
    else:
        new_w = int(scale * net_w)
        new_h = int(net_w / new_ar)

    dx = int(np.random.uniform(0, net_w - new_w))
    dy = int(np.random.uniform(0, net_h - new_h))

    return new_w, new_h, dx, dy


def random_scale_and_crop(image, new_h, new_w, net_h, net_w, dx, dy):
    """
             把图像先缩放到new_w和new_h大小，然后把图像的一边补上dx个像素，最后在另一边补像素直到net大小。dy同理

             Parameters
             ----------
             image : ndarray 原图
             new_h : int 缩放后的h
             new_w : int 缩放后的w
             net_h : int  输入网络的图像的h，常用的是416
             net_w : int 输入网络的图像的，常用的是416
             dx : int
             dy : int

             Returns
             -------
             img_adjusted[:net_h, :net_w, :] :

             Examples
             --------
             new_h = 451
             new_w = 276
             dx = 19
             dy = -23

            img_adjusted =
         """
    img_adjusted = cv2.resize(image, (new_w, new_h))

    if dx > 0:
        img_adjusted = np.pad(img_adjusted, ((0, 0), (dx, 0), (0, 0)), mode='constant', constant_values=127)
    else:
        img_adjusted = img_adjusted[:, -dx:, :]
    if (new_w + dx) < net_w:
        img_adjusted = np.pad(img_adjusted, ((0, 0), (0, net_w - (new_w + dx)), (0, 0)), mode='constant',
                              constant_values=127)

    if dy > 0:
        img_adjusted = np.pad(img_adjusted, ((dy, 0), (0, 0), (0, 0)), mode='constant', constant_values=127)
    else:
        img_adjusted = img_adjusted[-dy:, :, :]
    if (new_h + dy) < net_h:
        img_adjusted = np.pad(img_adjusted, ((0, net_h - (new_h + dy)), (0, 0), (0, 0)), mode='constant',
                              constant_values=127)
    return img_adjusted[:net_h, :net_w, :]


def correct_boxes(boxes, new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h):
    """
                 图像随机缩放、裁剪、翻转后，对应的box也要同样操作

                 Parameters
                 ----------
                 boxes : list
                 new_w : int 缩放后的h
                 new_h : int 缩放后的w
                 net_w : int  输入网络的图像的h，常用的是416
                 net_h : int 输入网络的图像的，常用的是416
                 dx : int
                 dy : int
                 flip : int 0或者1
                 image_w : 图像最原始的w
                 image_h : 图像最原始的h

                 Returns
                 -------
                 boxes :

                 Examples
                 --------
                 input boxes = [{'xmin': 696, 'ymin': 411, 'name': 'die knot', 'xmax': 1231, 'ymax': 565},
                                {'xmin': 696, 'ymin': 411, 'name': 'die knot', 'xmax': 1231, 'ymax': 565}]
                 output boxes = [{'ymax': 169, 'xmax': 263, 'ymin': 122, 'name': 'die knot', 'xmin': 194},
                                 {'ymax': 169, 'xmax': 263, 'ymin': 122, 'name': 'die knot', 'xmin': 194}]
            """

    def _constrain(min_v, max_v, value):
        if value < min_v:
            return min_v
        if value > max_v:
            return max_v
        return value

    boxes = copy.deepcopy(boxes)
    sx, sy = float(new_w) / image_w, float(new_h) / image_h

    for i in range(len(boxes)):
        boxes[i]['xmin'] = int(_constrain(0, net_w, boxes[i]['xmin'] * sx + dx))
        boxes[i]['xmax'] = int(_constrain(0, net_w, boxes[i]['xmax'] * sx + dx))
        boxes[i]['ymin'] = int(_constrain(0, net_h, boxes[i]['ymin'] * sy + dy))
        boxes[i]['ymax'] = int(_constrain(0, net_h, boxes[i]['ymax'] * sy + dy))

        if flip == 1:
            swap = boxes[i]['xmin']
            boxes[i]['xmin'] = net_w - boxes[i]['xmax']
            boxes[i]['xmax'] = net_w - swap

    return boxes


# TODO maybe list.pop is better than list.remove
def remove_outbox(boxes):
    temp = copy.deepcopy(boxes)
    for i, obj in enumerate(temp):
        if (obj['xmin'] == 416 and obj['xmax'] == 416) or (obj['ymin'] == 416 and obj['ymax'] == 416):
            boxes.remove(obj)
    return boxes


def remove_smallobj(boxes):
    temp = copy.deepcopy(boxes)
    for i, obj in enumerate(temp):
        if (obj['xmax'] - obj['xmin'] < 30) and (obj['ymax'] - obj['ymin'] < 30) or (
                (obj['xmax'] - obj['xmin']) * (obj['ymax'] - obj['ymin']) < 750):
            boxes.remove(obj)
    return boxes


def get_data(chunk, images_dir):
    """
            获得一张经过augement后的图像数据和经过同样操作的这张图片中所有box数据

            Parameters
            ----------
            chunk : list
            images_dir : str 训练图片所在文件夹

            Returns
            -------
            img_adjusted : ndarray [416,416,3] 0到255
            boxes_adjusted : list

            Examples
            --------
            chunk = ['1065.jpg', [2048, 1536, [{'name': 'die knot', 'ymax': 819, 'xmin': 1026, 'xmax': 1478, 'ymin': 482},
                                               {'name': 'die knot', 'ymax': 714, 'xmin': 796, 'xmax': 986, 'ymin': 387}]]]
            img_adjusted =
            boxes_adjusted = [{'xmax': 253, 'ymin': 159, 'ymax': 238, 'name': 'die knot', 'xmin': 101},
                           {'xmax': 87, 'ymin': 137, 'ymax': 214, 'name': 'die knot', 'xmin': 23}]
            """
    net_w = net_h = 416
    img_abs_path = images_dir + chunk[0]
    w, h, allobj_ = chunk[1]

    if allobj_ is None:
        return None, None
    image = cv2.imread(img_abs_path)  # RGB image

    if image is None:
        print('Cannot find ', img_abs_path)
    image = image[:, :, ::-1]  # RGB image
    image_h, image_w, _ = image.shape

    # apply scaling and cropping
    new_w, new_h, dx, dy = adjust_wh_ratio(image_h, image_w, net_h, net_w)
    img_adjusted = random_scale_and_crop(image, new_h, new_w, net_h, net_w, dx, dy)
    # randomly distort hsv space
    img_adjusted = random_distort_image(img_adjusted)
    # randomly flip
    flip = 0
    # flip = np.random.randint(2)
    # img_adjusted = random_flip(img_adjusted, flip)
    # correct the size and pos of bounding boxes
    boxes_adjusted = correct_boxes(allobj_, new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h)
    # remove the box which out of the 416*416 after augmentation
    # boxes_adjusted = remove_outbox(boxes_adjusted)
    # remove the box which ares is too small to get nan loss
    # boxes_adjusted = remove_smallobj(boxes_adjusted)
    return img_adjusted, boxes_adjusted


def bbox_iou(box1, box2):
    """
                计算两个box的iou，这里的2个box都是要以坐标左上角（0，0）点为起点，即box的xmin和ymin都要时0

                Parameters
                ----------
                box1 : list [xmin,ymin,xmax.ymax]
                box2 : list [xmin,ymin,xmax.ymax]

                Returns
                -------
                return = float

                Examples
                --------
                box1 = [0,0,89,76]
                box2 = [0,0,125,311]
                return = 0.002
                """

    def _interval_overlap(interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    intersect_w = _interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
    intersect_h = _interval_overlap([box1[1], box1[3]], [box2[1], box2[3]])

    intersect = intersect_w * intersect_h

    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def get_y_true(boxes):
    """
                将整个batch的boxes信息转换成最后的y_true形式

                Parameters
                ----------
                boxes : list

                Returns
                -------
                y_true = list [ndarray.shape[batch_size,52,52,3,6],
                               ndarray.shape[batch_size,26,26,3,6],
                               ndarray.shape[batch_size,13,13,3,6]]

                Examples
                --------
                boxes = [[{'xmin': 96, 'ymax': 285, 'ymin': 137, 'xmax': 231, 'name': 'box'}],
                        [{'xmin': 0, 'ymax': 315, 'ymin': 171, 'xmax': 152, 'name': 'box'}],
                        ...省略了14个，总共16个
                        ]
                y_true =
                """
    batch_size = Gb_batch_size
    base_grid_h = base_grid_w = 13
    net_w = net_h = 416
    anchors = Gb_anchors
    anchors_BoundBox = [[0, 0, anchors[2 * i], anchors[2 * i + 1]] for i in range(len(anchors) // 2)]
    labels = Gb_label

    y_true = list()
    # initialize the inputs and the outputs
    y_true.append(np.zeros((batch_size, 4 * base_grid_h, 4 * base_grid_w, 3,
                            4 + 1 + len(Gb_label))))  # desired network output 3
    y_true.append(np.zeros((batch_size, 2 * base_grid_h, 2 * base_grid_w, 3,
                            4 + 1 + len(Gb_label))))  # desired network output 2
    y_true.append(np.zeros((batch_size, 1 * base_grid_h, 1 * base_grid_w, 3,
                            4 + 1 + len(Gb_label))))  # desired network output 1
    for instance_index in range(batch_size):
        allobj_sized = boxes[instance_index]
        for obj in allobj_sized:
            # find the best anchor box for this object
            max_index = -1
            max_iou = -1
            shifted_box = [0, 0, obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin']]
            for i in range(len(anchors_BoundBox)):
                anchor = anchors_BoundBox[i]
                iou = bbox_iou(shifted_box, anchor)
                if max_iou < iou:
                    max_index = i
                    max_iou = iou
            # determine the yolo to be responsible for this bounding box
            grid_h, grid_w = y_true[max_index // 3].shape[1:3]
            # determine the position of the bounding box on the grid
            center_x = .5 * (obj['xmin'] + obj['xmax'])
            center_x = center_x / float(net_w)  # * grid_w  # sigma(t_x) + c_x
            center_y = .5 * (obj['ymin'] + obj['ymax'])
            center_y = center_y / float(net_h)  # * grid_h  # sigma(t_y) + c_y
            # determine the sizes of the bounding box
            w = (obj['xmax'] - obj['xmin']) / float(net_w)  # t_w
            h = (obj['ymax'] - obj['ymin']) / float(net_h)  # t_h
            box = [center_x, center_y, w, h]
            # determine the index of the label
            obj_indx = labels.index(obj['name'])
            # determine the location of the cell responsible for this object
            grid_x = int(np.floor(center_x * grid_w))
            grid_y = int(np.floor(center_y * grid_h))
            # assign ground truth x, y, w, h, confidence and class probs to y_batch
            y_true[max_index // 3][instance_index, grid_y, grid_x, max_index % 3] = 0
            y_true[max_index // 3][instance_index, grid_y, grid_x, max_index % 3, 0:4] = box
            y_true[max_index // 3][instance_index, grid_y, grid_x, max_index % 3, 4] = 1.
            y_true[max_index // 3][instance_index, grid_y, grid_x, max_index % 3, 5 + obj_indx] = 1
    return y_true


def data_generator(chunks, is_show=False):
    """
            根据epoch大小不断产生训练所需要的x_true，y_true

            Parameters
            ----------
            chunks : list
            is_show : bool 如果为True，可以显示训练前的图片

            Returns
            -------
            yield
                x_true : ndarray [batch_size,416,416,3]
                y_true : list [[batch_size,52,52,3,5 + n_class],
                               [batch_size,26,26,3,5 + n_class],
                               [batch_size,13,13,3,5 + n_class]]

            Examples
            --------
            chunks = [['1065.jpg', [2048, 1536, [{'name': 'die knot', 'ymax': 819, 'xmin': 1026, 'xmax': 1478, 'ymin': 482},
                                                {'name': 'die knot', 'ymax': 714, 'xmin': 796, 'xmax': 986, 'ymin': 387}]]],
                      ['10.jpg', [2048, 1536, [{'ymax': 1074, 'name': 'die knot', 'xmax': 982, 'xmin': 478, 'ymin': 535}]]]]
            x_true =
            y_true =
        """
    images_path = Gb_data_dir + 'images/'
    batch_size = Gb_batch_size
    n = len(chunks)
    i = 0
    count = 0
    while count < (n / Gb_batch_size):
        x_true = []
        box_data = []
        while len(box_data) < batch_size:
            i %= n
            imgs_sized, boxes_sized = get_data(chunks[i], images_path)
            i += 1
            if is_show == True:
                plt.cla()
                plt.imshow(imgs_sized)
                for obj in boxes_sized:
                    x1 = obj['xmin']
                    x2 = obj['xmax']
                    y1 = obj['ymin']
                    y2 = obj['ymax']

                    plt.hlines(y1, x1, x2, colors='red')
                    plt.hlines(y2, x1, x2, colors='red')
                    plt.vlines(x1, y1, y2, colors='red')
                    plt.vlines(x2, y1, y2, colors='red')
                plt.show()

            if len(boxes_sized) is 0:  # in case all the box in a batch become empty becase of the augmentation
                continue

            x_true.append(imgs_sized)
            box_data.append(boxes_sized)
        y_true = get_y_true(box_data)

        x_true = np.array(x_true)
        x_true = x_true / 255.

        yield x_true, y_true
        count += 1


if __name__ == '__main__':
    pick = Gb_label
    chunks = read_xml('train.txt', pick)
    generator = data_generator(chunks, is_show=True)
    for x_true, y_true in generator:
        print('ok')

    exit()
