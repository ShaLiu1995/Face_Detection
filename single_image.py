import tensorflow as tf
import numpy as np
import os
from scipy.misc import imread, imresize, imsave
from vgg16 import vgg16
from train_fc import final_fc, evaluate_bbx, draw_bbx

STD_SIZE = 224


def generate_feature(path):
    with tf.Session() as sess:
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

        print(file)
        img = imread(path, mode='RGB')
        print(img.shape)
        img = imresize(img, (STD_SIZE, STD_SIZE))

        feature = sess.run(vgg.middle, feed_dict={vgg.imgs: [img]})
        dim = feature.shape
        feature_array = np.reshape(feature, (dim[0], dim[1] * dim[2] * dim[3]))

        test_data = feature_array.reshape(1, -1)

        return test_data


if __name__ == '__main__':
    folder = 'prof_img'
    file = 'prof.png'
    old_path = os.path.join(folder, file)
    new_path = os.path.join(folder, 'labelled_' + file)

    test_data = generate_feature(old_path)
    model = final_fc()
    model.load_weights(os.path.join('saved_model', 'fc_model'))

    bbx = model.predict(test_data)[0]
    print(bbx)
    path = os.path.join(folder, file)
    draw_bbx(old_path, new_path, bbx)

