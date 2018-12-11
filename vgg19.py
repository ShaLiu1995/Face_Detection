import tensorflow as tf
import numpy as np
import os
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from imagenet_classes import class_names


def _conv_layer(input,weight,bias):
    conv = tf.nn.conv2d(input,weight,strides=[1,1,1,1],padding="SAME")
    return tf.nn.bias_add(conv,bias)


def _pool_layer(input):
    return tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


def preprocess(image,mean_pixel):
    return image-mean_pixel


def unprocess(image,mean_pixel):
    return image+mean_pixel


def imread(path):
    return scipy.misc.imread(path)


def imsave(image,path):
    img = np.clip(image,0,255).astype(np.int8)
    scipy.misc.imsave(path,image)


def net(data_path, input_image, sess=None):
    """
    :param data_path: VGG model directory
    :param input_image: image for testing
    :return:
    """
    layers = (
        'conv1_1', 'conv1_2', 'pool1',
        'conv2_1', 'conv2_2', 'pool2',
        'conv3_1', 'conv3_2', 'conv3_3','conv3_4', 'pool3',
        'conv4_1', 'conv4_2', 'conv4_3','conv4_4', 'pool4',
        'conv5_1', 'conv5_2', 'conv5_3','conv5_4', 'pool5',
        'fc1',     'fc2',     'fc3',
        'softmax'
    )
    data = scipy.io.loadmat(data_path)
    mean = data["normalization"][0][0][0][0][0]
    input_image = np.array([preprocess(input_image, mean)]).astype(np.float32)
    net = {}
    current = input_image
    net["src_image"] = tf.constant(current)
    count = 0
    for i in range(43):
        if str(data['layers'][0][i][0][0][0][0])[:4] == ("relu"):
            continue
        if str(data['layers'][0][i][0][0][0][0])[:4] == ("pool"):
            current = _pool_layer(current)
        elif str(data['layers'][0][i][0][0][0][0]) == ("softmax"):
            current = tf.nn.softmax(current)
        elif i == (37):
            shape = int(np.prod(current.get_shape()[1:]))
            current = tf.reshape(current, [-1, shape])
            kernels, bias = data['layers'][0][i][0][0][0][0]
            kernels = np.reshape(kernels,[-1,4096])
            bias = bias.reshape(-1)
            current = tf.nn.relu(tf.add(tf.matmul(current,kernels),bias))
        elif i == (39):
            kernels, bias = data['layers'][0][i][0][0][0][0]
            kernels = np.reshape(kernels,[4096,4096])
            bias = bias.reshape(-1)
            current = tf.nn.relu(tf.add(tf.matmul(current,kernels),bias))
        elif i == 41:
            kernels, bias = data['layers'][0][i][0][0][0][0]
            kernels = np.reshape(kernels, [4096, 1000])
            bias = bias.reshape(-1)
            current = tf.add(tf.matmul(current, kernels), bias)
        else:
            kernels,bias = data['layers'][0][i][0][0][0][0]
            # kernels = np.transpose(kernels,[1,0,2,3])
            bias = bias.reshape(-1)
            current = tf.nn.relu(_conv_layer(current,kernels,bias))
        net[layers[count]] = current
        count += 1
    return net, mean


if __name__ == '__main__':
    VGG_PATH = os.getcwd()+"/imagenet-vgg-verydeep-19.mat"

    input_image = scipy.misc.imread("test_face.jpg")
    input_image = scipy.misc.imresize(input_image,[224,224,3])

    shape = (1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
    # image = tf.placeholder('float', shape=shape)

    with tf.Session() as sess:
        nets, mean_pixel, = net(VGG_PATH, input_image, sess=sess)
        # print(sess.run(nets,feed_dict={image:input_image}))
        nets = sess.run(nets)

        # for key, values in nets.items():
        #     if len(values.shape) < 4:
        #         continue
        #     plt.figure(key)
        #     plt.matshow(values[0, :, :, 0],)
        #     plt.title(key)
        #     plt.colorbar()
        #     plt.show()

        net_sort = list(reversed(np.argsort(nets["softmax"]).reshape(-1).tolist()))
        net_softmax = nets["softmax"].reshape(-1).tolist()
        for i in range(10):
            print(class_names[net_sort[i]], ": ", net_softmax[net_sort[i]])
