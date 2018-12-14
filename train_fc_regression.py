import numpy as np
import cv2
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from scipy.misc import imread, imsave, imresize
from eval_util import evaluate_test_set, get_iou, get_acc


# def final_fc(weights_path=None):
#     model = Sequential()
#     model.add(Dense(4096, activation='relu', input_dim=25088))
#     model.add(Dense(4096, activation='relu'))  
#     model.add(Dense(4, activation='relu'))
#     if weights_path:
#         model.load_weights(weights_path)
#     return model


def final_fc(weights_path=None):
    model = Sequential()
    model.add(Dense(8192, activation='relu', input_dim=25088))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='relu'))
    if weights_path:
        model.load_weights(weights_path)
    return model


def train_fc(xtrain, ltrain):
    print('Training data size: {}'.format(xtrain.shape))
    print('Training label size: {}'.format(ltrain.shape))
    model = final_fc()
    my_sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    # my_sgd = SGD(lr=0.001)
    model.compile(optimizer=my_sgd, loss='mse')

    model.fit(xtrain, ltrain, epochs=5, batch_size=128)
    print('Training error: {}'.format(model.evaluate(xtrain, ltrain)))
    model.save_weights(os.path.join('saved_model', 'fc_regression'))
    return model


if __name__ == "__main__":
    TRAIN_SIZE = 4600 * 4
    TEST_SIZE = 1100

    TRAIN_FEATURE_FILE = 'feature_array.bin'
    TEST_FEATURE_FILE =  'feature_array_val.bin'
#     TRAIN_FEATURE_FILE = 'shuffled_25088.bin'
#     TEST_FEATURE_FILE =  'test_25088.bin'
    TRAIN_BBX_FILE = 'bbx_array.bin'
    TEST_BBX_FILE = 'bbx_array_val.bin' 

    xtrain = np.fromfile(TRAIN_FEATURE_FILE, dtype=np.float32).reshape(TRAIN_SIZE, -1)
    ltrain = np.fromfile(TRAIN_BBX_FILE, dtype=np.int64).reshape(TRAIN_SIZE, -1)
    
    xtest = np.fromfile(TEST_FEATURE_FILE, dtype=np.float32).reshape(TEST_SIZE, -1)
    ltest = np.fromfile(TEST_BBX_FILE, dtype=np.int64).reshape(TEST_SIZE, -1)
    
#     print(xtrain.shape)
#     print(ltrain.shape)
#     print(xtest.shape)
#     print(ltest.shape)
        

#     xdata_split = np.vsplit(xdata, 8)
#     ldata_split = np.vsplit(ldata, 8)
    
#     xtrain = np.vstack((xdata_split[0], xdata_split[2], xdata_split[4], xdata_split[6]))
#     xtest = np.vstack((xdata_split[1], xdata_split[3], xdata_split[5], xdata_split[7]))
    
#     ltrain = np.vstack((ldata_split[0], ldata_split[2], ldata_split[4], ldata_split[6]))
#     ltest = np.vstack((ldata_split[1], ldata_split[3], ldata_split[5], ldata_split[7]))
       
#     xtrain = xdata[:TRAIN_SIZE, :]
#     ltrain = ldata[:TRAIN_SIZE, :]
#     xtest = xdata[TRAIN_SIZE:, :]
#     ltest = ldata[TRAIN_SIZE:, :]

    model = train_fc(xtrain, ltrain)
    print('Testing error: {}'.format(model.evaluate(xtest, ltest)))