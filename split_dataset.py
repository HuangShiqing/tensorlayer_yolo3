import os
import numpy
import random
import numpy as np
def create_path(project_dir = './',annotation_dir = project_dir+'label/',img_dir = project_dir+'image/',train_percent=0.8,val_percent = 0.2,test_percent = 0):
    img_list_anno = []
    img_list_others = []
    
    compare_list = {}
    for i in os.listdir('./image/'):
        compare_list[i.split('.')[0]] = i.split('.')[1]
    
    for i in os.listdir(annotation_dir):
        img_list_anno.append(img_dir+i.split('.')[0]+'.'+compare_list[i.split('.')[0]])

    for j in os.listdir(img_dir):
        if((img_dir+j) not in img_list_anno):
            img_list_others.append(img_dir+j)
    np.random.seed(10101)
    np.random.shuffle(img_list_anno)
    num_val = int(len(img_list_anno)*val_percent)
    num_test = int(len(img_list_anno)*test_percent)
    num_train = len(img_list_anno) - num_val-num_test

    with open(project_dir+'train.txt','a') as f:
        for item in img_list_anno[:num_train]:
            f.write('%s\n' % item)
    with open(project_dir+'val.txt','a') as f:
        for item in img_list_anno[num_train:(num_train+num_val)]:
            f.write('%s\n' % item)
    with open(project_dir+'test.txt','a') as f:
        for item in img_list_anno[(num_train+num_val):]:
            f.write('%s\n' % item)

    if(img_list_others !=[]):
        with open(project_dir+'others.txt','a') as f:
            for item in img_list_others:
                f.write('%s\n' % item)

if __name__ == "__main__":
    create_path()
