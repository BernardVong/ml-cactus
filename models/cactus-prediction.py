import os
import cv2
import numpy as np
import pandas as pd
# MacOS : fix compatibility
from keras.engine.saving import model_from_json
from keras.utils import np_utils
from sklearn.model_selection import train_test_split



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# GLOBALS
PROJECT_PATH = os.getcwd().replace("/models", "")
filename = ''


def imagesToDataset(paths):
    data = []
    for filepath in paths:
        file = cv2.imread(filepath).astype(np.float32)/255
        data.append(file)
    return np.array(data)


# load dataset
files = [PROJECT_PATH + '/data/test/']
dataset = imagesToDataset(files)


# load model json
json_file = open(filename + '.json', 'r')
model_json = json_file.read()
json_file.close()

# load model weights
model = model_from_json(model_json)
model.load_weights(filename + ".h5")


model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = model.evaluate(dataset, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
