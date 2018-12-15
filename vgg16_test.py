
import tensorflow as tf
import numpy as np
import os
import json
import csv
import logging
from scipy.misc import imread, imresize
from flip_data import flip_bbx
from vgg16 import vgg16


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    image_dir_list = ['resized_img_test_easy']
    BATCH_SIZE = 50
    TEST_SIZE = 3000

    with open(os.path.join('json_files', 'bbx_test_dict.json')) as json_data:
        bbx_dict = json.load(json_data)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg = vgg16(imgs, os.path.join('saved_model', 'vgg16_weights.npz'), sess)
     
        for k in range(len(image_dir_list)):
            
            bbx_array = np.empty((0, 4), dtype=np.int64)
            feature_array = np.empty((0, 25088), dtype=np.float32)
        
            image_dir = image_dir_list[k] 
            file_list = os.listdir(image_dir)[:TEST_SIZE]
            file_num = len(file_list)
            batch_num = int(file_num / BATCH_SIZE)
            print('{0} image(s), {1} batch(es)'.format(file_num, batch_num))
              
            for i in range(batch_num):
                logging.debug('Reading folder No.{0}, batch No.{1}'.format(k+1, i+1))
                img_batch_list = []
                for j in range(BATCH_SIZE):
                    # logging.debug('Batch No. {0}, image No. {1}'.format(i, j))
                    idx = i * BATCH_SIZE + j
                    file = file_list[idx]
    
                    img = imread(os.path.join(image_dir, file), mode='RGB')
                    img_batch_list.append(img)
    
                    bbx = np.array(np.array(bbx_dict[file])).reshape(1, -1)
                    bbx[0] = flip_bbx(bbx[0], flag=k)
                    bbx_array = np.append(bbx_array, bbx, axis=0)
    
                feature = sess.run(vgg.middle, feed_dict={vgg.imgs: img_batch_list})
                feature = feature.reshape(feature.shape[0], -1)
                feature_array = np.append(feature_array, feature, axis=0)

            print('Bounding box array shape: {}'.format(bbx_array.shape))
            print('Feature array shape: {}'.format(feature_array.shape))
              
            bbx_array.tofile(os.path.join('bin_files', 'bbx_array_test_easy.bin'))
            feature_array.tofile(os.path.join('bin_files', 'feature_array_test_easy.bin'))
