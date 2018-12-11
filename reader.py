from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.io import loadmat
from scipy.misc import imread, imresize, imsave
import os
import logging
import cv2
import shutil

SIZE = 224


def resize_bbx(img, bbox):
    print(img.shape)
    height = img.shape[0]
    width = img.shape[1]
    new_bbox = [0] * 4
    new_bbox[0] = int(SIZE * bbox[0][0][0] / width)
    new_bbox[1] = int(SIZE * bbox[0][0][1] / height)
    new_bbox[2] = int(SIZE * bbox[0][0][2] / width)
    new_bbox[3] = int(SIZE * bbox[0][0][3] / height)
    return new_bbox


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    TRAIN_DIR = os.path.join('WIDER_train', 'images')
    NEW_IMG_DIR = 'resized_data'
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

            img = imread(img_path)
            # Plot bbx with original image
            # cv2.rectangle(img, (bbx[0][0][0], bbx[0][0][1]), (bbx[0][0][0] + bbx[0][0][2], bbx[0][0][1] + bbx[0][0][3]),
            #           (255, 0, 0), 2)
            # imsave(os.path.join(NEW_IMG_DIR, filename + '.jpg'), img)

            new_img = imresize(img, (224, 224))
            xtrain.append(new_img)

            logging.debug('Image Size: {}\n'.format(xtrain[len(xtrain) - 1].shape))
            ltrain_bbx.append(bbx)

            new_bbx = resize_bbx(img, bbx)

            cv2.rectangle(new_img, (new_bbx[0], new_bbx[1]), (new_bbx[0] + new_bbx[2], new_bbx[1] + new_bbx[3]), (255, 0, 0), 2)
            imsave(os.path.join(NEW_IMG_DIR, filename + '.jpg'), new_img)

    print(len(xtrain))
    print(len(ltrain_bbx))


