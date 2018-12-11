from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.io import loadmat
from scipy.ndimage import imread
import os
import logging


logging.basicConfig(level=logging.DEBUG)

TRAIN_DIR = os.path.join('WIDER_train', 'images')
LABEL_DIR = 'wider_face_split'
LTRAIN_FILE = 'wider_face_train.mat'
LVALID_FILE = 'wider_face_val.mat'
LTEST_FILE = 'wider_face_test.mat'
EVENT_NUM = 61

ltrain_mat = loadmat(os.path.join(LABEL_DIR, LTRAIN_FILE))
# lvalid = loadmat(os.path.join(LABEL_DIR, LVALID_FILE))
# ltest = loadmat(os.path.join(LABEL_DIR, LTEST_FILE))

ltrain_bbx_list = ltrain_mat['face_bbx_list']
ltrain_event_list = ltrain_mat['event_list']
ltrain_file_list = ltrain_mat['file_list']
ltrain_bbx = []
xtrain = []

for i in range(EVENT_NUM):
    event = ltrain_event_list[i][0][0]
    for j in range(len(ltrain_bbx_list[i][0])):
    # for j in range(50):
        bbx = ltrain_bbx_list[i][0][j]
        num_of_faces = bbx[0].shape[0]
        if num_of_faces > 1:
            continue
        filename = ltrain_file_list[i][0][j][0][0]
        img_path = os.path.join(TRAIN_DIR, event, filename + '.jpg')
        logging.debug('Path: {}'.format(img_path))
        logging.debug('BBox: {}'.format(str(bbx)))

        xtrain.append(imread(img_path))
        logging.debug('Image Size: {}\n'.format(xtrain[len(xtrain) - 1].shape))
        ltrain_bbx.append(bbx)


print(len(xtrain))
print(len(ltrain_bbx))
