import os
import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import copy


images_path = "D:/DeepLearning/data/VOCdevkit/VOC2012/JPEGImages/"
annotations_path = "D:/DeepLearning/data/VOCdevkit/VOC2012/Annotations/"


# images_name = os.listdir(images_path)
# abs_name = images_path + images_name[0]


# images_name = os.path.abspath(images_name)


# ann_name = os.listdir(annotations_path)
# abs_ann = annotations_path + ann_name[0]


# <class 'list'>: ['2007_000027.jpg', [486, 500, [['person', 174, 101, 349, 351]]]]
def pascal_voc_clean_xml(ANN, pick, exclusive=False):
    print('Parsing for {} {}'.format(
        pick, 'exclusively' * int(exclusive)))

    dumps = list()
    cur_dir = os.getcwd()
    os.chdir(ANN)
    annotations = os.listdir('.')
    # annotations = glob.glob(str(annotations) + '*.xml')
    size = len(annotations)

    for i, file in enumerate(annotations):
        # progress bar
        sys.stdout.write('\r')
        percentage = 1. * (i + 1) / size
        progress = int(percentage * 20)
        bar_arg = [progress * '=', ' ' * (19 - progress), percentage * 100]
        bar_arg += [file]
        sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
        sys.stdout.flush()

        # actual parsing
        in_file = open(file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        jpg = str(root.find('filename').text)
        imsize = root.find('size')
        w = int(imsize.find('width').text)
        h = int(imsize.find('height').text)
        all = list()

        for obj in root.iter('object'):
            # current = list()
            current = dict()
            name = obj.find('name').text
            if name not in pick:
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
        dumps += add
        in_file.close()

    # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current['name'] in pick:
                if current['name'] in stat:
                    stat[current['name']] += 1
                else:
                    stat[current['name']] = 1

    print('\nStatistics:')
    # _pp(stat)
    for i in stat: print('{}: {}'.format(i, stat[i]))
    print('Dataset size: {}'.format(len(dumps)))

    os.chdir(cur_dir)
    return dumps

def _rand_scale(scale):
    scale = np.random.uniform(1, scale)
    return scale if (np.random.randint(2) == 0) else 1./scale;

def random_flip(image, flip):
    if flip == 1: return cv2.flip(image, 1)
    return image

def _constrain(min_v, max_v, value):
    if value < min_v: return min_v
    if value > max_v: return max_v
    return value

def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5):
    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = _rand_scale(saturation);
    dexp = _rand_scale(exposure);

    # convert RGB space to HSV space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')

    # change satuation and exposure
    image[:, :, 1] *= dsat
    image[:, :, 2] *= dexp

    # change hue
    image[:, :, 0] += dhue
    image[:, :, 0] -= (image[:, :, 0] > 180) * 180
    image[:, :, 0] += (image[:, :, 0] < 0) * 180

    # convert back to RGB from HSV
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)


def apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy):
    im_sized = cv2.resize(image, (new_w, new_h))

    if dx > 0:
        im_sized = np.pad(im_sized, ((0, 0), (dx, 0), (0, 0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[:, -dx:, :]
    if (new_w + dx) < net_w:
        im_sized = np.pad(im_sized, ((0, 0), (0, net_w - (new_w + dx)), (0, 0)), mode='constant', constant_values=127)

    if dy > 0:
        im_sized = np.pad(im_sized, ((dy, 0), (0, 0), (0, 0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[-dy:, :, :]

    if (new_h + dy) < net_h:
        im_sized = np.pad(im_sized, ((0, net_h - (new_h + dy)), (0, 0), (0, 0)), mode='constant', constant_values=127)

    return im_sized[:net_h, :net_w, :]

def correct_bounding_boxes(boxes, new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h):
    boxes = copy.deepcopy(boxes)

    # randomize boxes' order
    # np.random.shuffle(boxes)

    # correct sizes and positions
    sx, sy = float(new_w)/image_w, float(new_h)/image_h
    zero_boxes = []

    for i in range(len(boxes)):
        boxes[i]['xmin'] = int(_constrain(0, net_w, boxes[i]['xmin'] * sx + dx))
        boxes[i]['xmax'] = int(_constrain(0, net_w, boxes[i]['xmax'] * sx + dx))
        boxes[i]['ymin'] = int(_constrain(0, net_h, boxes[i]['ymin'] * sy + dy))
        boxes[i]['ymax'] = int(_constrain(0, net_h, boxes[i]['ymax'] * sy + dy))

        # if boxes[i]['xmax'] <= boxes[i]['xmin'] or boxes[i]['ymax'] <= boxes[i]['ymin']:
        #     zero_boxes += [i]
        #     continue

        if flip == 1:
            swap = boxes[i]['xmin'];
            boxes[i]['xmin'] = net_w - boxes[i]['xmax']
            boxes[i]['xmax'] = net_w - swap

    # boxes = [boxes[i] for i in range(len(boxes)) if i not in zero_boxes]

    return boxes


# a = pascal_voc_clean_xml(annotations_path, "person")
# chunk = a[0]

# chunk = ['2007_000027.jpg', [486, 500, [{'ymax': 351, 'name': 'person', 'xmax': 349, 'ymin': 101, 'xmin': 174}]]]
chunk = ['2007_000032.jpg', [500, 281, [{'name': 'person', 'ymax': 229, 'ymin': 180, 'xmin': 195, 'xmax': 213}, {'name': 'person', 'ymax': 238, 'ymin': 189, 'xmin': 26, 'xmax': 44}]]]
# chunk = ['2007_000027.jpg', [486, 500, [['person', 174, 101, 349, 351]]]]



net_w = net_h = 416
jitter = 0.3
img_abs_path = images_path + chunk[0]
w, h, allobj_ = chunk[1]

image = cv2.imread(img_abs_path)  # RGB image


# for obj in allobj_:
#     cv2.rectangle(image, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']),(0,255,0), 1)
# cv2.imwrite("C:/Users/john/Desktop/0.jpg", image)

if image is None: print('Cannot find ', img_abs_path)
image = image[:, :, ::-1]  # RGB image

image_h, image_w, _ = image.shape

# determine the amount of scaling and cropping
dw = jitter * image_w;
dh = jitter * image_h;

new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh));
scale = np.random.uniform(0.25, 2);

if (new_ar < 1):
    new_h = int(scale * net_h);
    new_w = int(net_h * new_ar);
else:
    new_w = int(scale * net_w);
    new_h = int(net_w / new_ar);

dx = int(np.random.uniform(0, net_w - new_w));
dy = int(np.random.uniform(0, net_h - new_h));

# apply scaling and cropping
im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)
# randomly distort hsv space
im_sized = random_distort_image(im_sized)
# randomly flip
flip = np.random.randint(2)
im_sized = random_flip(im_sized, flip)

# correct the size and pos of bounding boxes
allobj_sized = correct_bounding_boxes(allobj_, new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h)


for obj in allobj_sized:
    cv2.rectangle(im_sized,(obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']),(0,255,0), 1)
im_sized = im_sized[:, :, ::-1]  # RGB image
cv2.imwrite("C:/Users/john/Desktop/1.jpg", im_sized)















tim = 0
