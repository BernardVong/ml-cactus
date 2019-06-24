# fix Mac problem
import os
import time
from random import randint

import cv2
import numpy as np
import pandas as pd
from keras import Model
from keras.callbacks import TensorBoard
from keras.layers import Dense, Flatten, Activation, BatchNormalization, Input, Add, AveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import sgd
from keras.regularizers import l2
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
num_classes = 2

project_path = os.getcwd().replace("/models", "")

''' DATA PROCESSING '''
def imagesToDataset(paths):
    data = []
    for filepath in paths:
        file = cv2.imread(filepath).astype(np.float32)/255
        data.append(file)
    return np.array(data)

train_csv = pd.read_csv(project_path + '/data/train.csv')
files = [project_path + '/data/train/' + file for file in train_csv['id'].tolist()]
trains = imagesToDataset(files)
labels = np_utils.to_categorical(train_csv['has_cactus'].tolist(), 2)


x_train, x_test, y_train, y_test = train_test_split(trains, labels, test_size=0.33)
''' DATA PROCESSING END '''



def resnet_layer(inputs, with_activation=True, num_filters=16):

    convolution = Conv2D(
        filters=num_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-4),
    )

    new_layer = convolution(inputs)
    new_layer = BatchNormalization()(new_layer)
    if with_activation:
        new_layer = Activation('relu')(new_layer)
    return new_layer


def model_builder(model_params):

    inputs = Input(shape=x_train.shape[1:])
    initial_layer = resnet_layer(inputs=inputs)

    layers = add_block(initial_layer)
    for block_number in range(model_params['blocks']):
        layers = add_block(initial_layer if block_number == 0 else layers)

    layers = AveragePooling2D(pool_size=8)(layers)
    layers_flatten = Flatten()(layers)
    outputs = Dense(num_classes,
                    activation=model_params['activation'],
                    kernel_initializer='he_normal')(layers_flatten)

    # Instantiate model.
    new_model = Model(inputs=inputs, outputs=outputs)

    return new_model


def add_block(input_layer):
    blocks = resnet_layer(inputs=input_layer)
    blocks = resnet_layer(inputs=blocks)
    blocks = resnet_layer(inputs=blocks, with_activation=False)

    layers = Add()([input_layer, blocks])
    layers = Activation('relu')(layers)
    return layers


if __name__ == '__main__':

    for test_number in range(150):
        params = {
            'loss': 'mse',
            'activation': 'softmax',
            'batch_size': randint(1, 40),
            'epochs': 30,
            # lr : 0.05 to 0.08 best case
            'lr': 0.06,
            # mm : 0.80 to 0.85 best case
            'momentum': 0.85,
            'blocks': 5

            # round(random.uniform(1.5, 1.9), 2)

        }

        model = model_builder(params)

        optimizer = sgd(
            lr=params['lr'],
            momentum=params['momentum'],
            decay=0,
            nesterov=False
        )

        model.compile(
            loss=params['loss'],
            optimizer=optimizer,
            metrics=['accuracy']
        )
        model.summary()

        # tensorflow config
        model_label = "resnet_" + params['activation'] + "_sgd_blocks" + str(params['blocks']) + "_lr" + str(
            params['lr']) + "_mm" + str(
            params['momentum']) + "_" + params['loss'] + '_bs' + str(params['batch_size']) + '_' + str(round(time.time()))
        tb_callback = TensorBoard(log_dir=project_path + "/logs/" + model_label)

        # fit model
        model_fitted = model.fit(
            x_train,
            y_train,
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            validation_data=(x_test, y_test),
            shuffle=False,
            callbacks=[tb_callback]
        )
        prediction = model.predict(x_test)
