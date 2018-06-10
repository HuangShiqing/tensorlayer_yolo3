import numpy as np


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union



batch_size = 4
base_grid_h = base_grid_w = 13
net_w = net_h = 416

anchors = [55, 69, 75, 234, 133, 240, 136, 129, 142, 363, 203, 290, 228, 184, 285, 359, 341, 260]
labels = ['car', 'person']


anchors_BoundBox = [BoundBox(0, 0, anchors[2*i], anchors[2*i+1]) for i in range(len(anchors)//2)]

# initialize the inputs and the outputs
yolo_1 = np.zeros((batch_size, 1 * base_grid_h, 1 * base_grid_w, len(anchors_BoundBox) // 3,
                   4 + 1 + len(labels)))  # desired network output 1
yolo_2 = np.zeros((batch_size, 2 * base_grid_h, 2 * base_grid_w, len(anchors_BoundBox) // 3,
                   4 + 1 + len(labels)))  # desired network output 2
yolo_3 = np.zeros((batch_size, 4 * base_grid_h, 4 * base_grid_w, len(anchors_BoundBox) // 3,
                   4 + 1 + len(labels)))  # desired network output 3
yolos = [yolo_3, yolo_2, yolo_1]

instance_count = 0
true_box_index = 0

# for train_instance in self.instances[l_bound:r_bound]:
for i in range(batch_size):

    im_sized = np.zeros([net_w, net_h, 3])
    allobj_sized = [{'xmin': 254, 'name': 'person', 'ymin': 260, 'xmax': 262, 'ymax': 325},
                    {'xmin': 329, 'name': 'person', 'ymin': 272, 'xmax': 337, 'ymax': 337}]

    for obj in allobj_sized:
        # find the best anchor box for this object
        max_anchor = None
        max_index = -1
        max_iou = -1

        shifted_box = BoundBox(0,
                               0,
                               obj['xmax'] - obj['xmin'],
                               obj['ymax'] - obj['ymin'])

        for i in range(len(anchors_BoundBox)):
            anchor = anchors_BoundBox[i]
            iou = bbox_iou(shifted_box, anchor)

            if max_iou < iou:
                max_anchor = anchor
                max_index = i
                max_iou = iou

                # determine the yolo to be responsible for this bounding box
        yolo = yolos[max_index // 3]
        grid_h, grid_w = yolo.shape[1:3]

        # determine the position of the bounding box on the grid
        center_x = .5 * (obj['xmin'] + obj['xmax'])
        center_x = center_x / float(net_w) * grid_w  # sigma(t_x) + c_x
        center_y = .5 * (obj['ymin'] + obj['ymax'])
        center_y = center_y / float(net_h) * grid_h  # sigma(t_y) + c_y

        # determine the sizes of the bounding box
        w = np.log((obj['xmax'] - obj['xmin']) / float(max_anchor.xmax))  # t_w
        h = np.log((obj['ymax'] - obj['ymin']) / float(max_anchor.ymax))  # t_h

        box = [center_x, center_y, w, h]

        # determine the index of the label
        obj_indx = labels.index(obj['name'])

        # determine the location of the cell responsible for this object
        grid_x = int(np.floor(center_x))
        grid_y = int(np.floor(center_y))

        # assign ground truth x, y, w, h, confidence and class probs to y_batch
        yolo[instance_count, grid_y, grid_x, max_index % 3] = 0
        yolo[instance_count, grid_y, grid_x, max_index % 3, 0:4] = box
        yolo[instance_count, grid_y, grid_x, max_index % 3, 4] = 1.
        yolo[instance_count, grid_y, grid_x, max_index % 3, 5 + obj_indx] = 1