import os
import time

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.activations import sigmoid
from keras.callbacks import TensorBoard
from keras.layers import Dense, MaxPooling2D, Dropout, Flatten
from keras.losses import mse, binary_crossentropy
from keras.models import Sequential
from keras.optimizers import sgd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# MacOS : fix compatibility
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


params = {
    'num_classes': 2,
    'batch_size': 32,
    'epochs': 50,
    'lr': 0.01,
    'momentum': 0.9,
    'buffer_size': 10000,
    'image_size': 32
}



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


# create and train model
model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(params['num_classes'], activation=sigmoid))

model.compile(loss=binary_crossentropy,
              optimizer=sgd(lr=params['lr']),
              metrics=['accuracy'])


model.summary()

# tensorflow config
model_logs = project_path + "/logs/" + "mlp" + str(4) + "_" + "softmax_sgd_" + str(params['lr']) + '_' + str(round(time.time()))
tb_callback = TensorBoard(log_dir=model_logs)


# Fit model
modelFitted = model.fit(
    x_train, y_train,
    batch_size=params['batch_size'],
    epochs=params['epochs'],
    validation_data=(x_test, y_test),
    callbacks=[tb_callback]
)
