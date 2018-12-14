import os
import numpy as np
import json
import cv2
from scipy.misc import imread, imresize, imsave


def flip_img(old_dir, flip):
    new_dir = '{0}_{1}'.format(old_dir, flip)
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    file_list = os.listdir(old_dir)
    for file in file_list:
        img = imread(os.path.join(old_dir, file), mode='RGB')
        img = img.copy()
        # 0: ud, 1: lr
        flag = 0
        if flip == 'ud':
            img = np.flip(img, 0)
            flag = 1
        elif flip == 'lr':
            img = np.flip(img, 1)
            flag = 2
        elif flip == 'udlr':
            img = np.flip(img, 0)
            img = np.flip(img, 1)
            flag = 3

        old_bbx = bbx_dict[file]
        new_bbx = flip_bbx(old_bbx, flag=flag)
        img = img.copy()
#         cv2.rectangle(img, (new_bbx[0], new_bbx[1]),
#                       (new_bbx[0] + new_bbx[2], new_bbx[1] + new_bbx[3]), (255, 0, 0), 2)
        imsave(os.path.join(new_dir, file), img)

    print('Finish flipping task {}'.format(flip))


def flip_bbx(old_bbx, flag=0):
    new_bbx = list(old_bbx)
    if flag == 1:   # ud
        new_bbx[1] = 224 - old_bbx[1] - old_bbx[3]
    elif flag == 2:     # lr
        new_bbx[0] = 224 - old_bbx[0] - old_bbx[2]
    elif flag == 3:     # udlr
        new_bbx[0] = 224 - old_bbx[0] - old_bbx[2]
        new_bbx[1] = 224 - old_bbx[1] - old_bbx[3]
    else:
        pass    # flag = 0
    return new_bbx


if __name__ == '__main__':

    with open('bbx_idx_dict.json') as json_data:
        bbx_dict = json.load(json_data)

    old_dir = 'resized_img_shuffled'
    flip_img(old_dir, 'ud')
    flip_img(old_dir, 'lr')
    flip_img(old_dir, 'udlr')
