import os
import xml.etree.ElementTree as ET

ANN = 'D:/DeepLearning/data/LongWoodCutPickJpg/valid_label/'
save_dir = 'D:/DeepLearning/code/tensorlayer_yolo3/gt/'

cur_dir = os.getcwd()
os.chdir(ANN)
annotations = os.listdir('.')

for i, file in enumerate(annotations):
    in_file = open(file)
    tree = ET.parse(in_file)
    root = tree.getroot()
    jpg = str(root.find('filename').text) + '.jpg'

    txt_file = open(save_dir + file.rstrip('xml') + 'txt', 'w')
    for obj in root.iter('object'):
        # current = list()
        current = dict()
        name = obj.find('name').text

        xmlbox = obj.find('bndbox')
        xn = int(float(xmlbox.find('xmin').text))
        xx = int(float(xmlbox.find('xmax').text))
        yn = int(float(xmlbox.find('ymin').text))
        yx = int(float(xmlbox.find('ymax').text))

        txt_file.write('{0} {1} {2} {3} {4}'.format(name, xn, yn, xx, yx))
    txt_file.close()
os.chdir(cur_dir)