import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD

N_DIM = 25088


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


if __name__ == "__main__":
    # model = final_fc('vgg16_weights.h5')
    xtrain = np.fromfile('feature_array.bin', dtype=np.float32).reshape(4630, -1)
    ltrain = np.fromfile('bbx_array.bin', dtype=int).reshape(4630, -1)
    model = final_fc()
    my_sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=my_sgd, loss='mse')
    model.fit(xtrain, ltrain, epochs=20, batch_size=64)
