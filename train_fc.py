import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from scipy.misc import imread, imsave, imresize

N_DIM = 25088
STD_SIZE = 224

def final_fc(weights_path=None):
    model = Sequential()
    model.add(Dense(8192, activation='relu', input_dim=N_DIM))
    # model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(128, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='relu'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def evaluate_bbx(lo, hi, bbx):
    IMAGE_DIR = 'resized_img'
    NEW_IMG_DIR = 'test_img'
    file_list = os.listdir(IMAGE_DIR)

    for i in range(lo, hi):
        file = file_list[i]
        old_path = os.path.join(IMAGE_DIR, file)
        new_path = os.path.join(NEW_IMG_DIR, file)
        draw_bbx(old_path, new_path, bbx[i - lo])


def draw_bbx(old_path, new_path, bbx):
    img = imread(old_path)
    w = img.shape[0]
    h = img.shape[1]
    if w != STD_SIZE or h != STD_SIZE:
        img = imresize(img, (w, h))
    cv2.rectangle(img, (bbx[0], bbx[1]),
                  (bbx[0] + bbx[2], bbx[1] + bbx[3]), (255, 0, 0), 2)
    if w != STD_SIZE or h != STD_SIZE:
        img = imresize(img, (STD_SIZE, STD_SIZE))
    imsave(new_path, img)


def run_training(xtrain, ltrain):
    print('Training data size: {}'.format(xtrain.shape))
    print('Training label size: {}'.format(ltrain.shape))
    model = final_fc()
    my_sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    # my_sgd = SGD(lr=0.001)
    model.compile(optimizer=my_sgd, loss='mse')

    model.fit(xtrain, ltrain, epochs=15, batch_size=128)
    print('Training error: {}'.format(model.evaluate(xtrain, ltrain)))
    model.save_weights(os.path.join('saved_model', 'fc'))
    return model


if __name__ == "__main__":
    TRAIN_SIZE = 4000
    TOTAL_SZIE = 4630

    xdata = np.fromfile('feature_array.bin', dtype=np.float32).reshape(TOTAL_SZIE, -1)
    ldata = np.fromfile('bbx_array.bin', dtype=np.int64).reshape(TOTAL_SZIE, -1)

    xtrain = xdata[:TRAIN_SIZE, :]
    ltrain = ldata[:TRAIN_SIZE, :]
    xtest = xdata[TRAIN_SIZE:, :]
    ltest = ldata[TRAIN_SIZE:, :]

    model = run_training(xtrain, ltrain)
    print('Testing error: {}'.format(model.evaluate(xtest, ltest)))
    predict_bbx = model.predict(xtest)
    evaluate_bbx(TRAIN_SIZE, TOTAL_SZIE, predict_bbx)
