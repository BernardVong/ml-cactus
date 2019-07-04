import os
import cv2
import numpy as np
import pandas as pd
# MacOS : fix compatibility
from keras.backend import binary_crossentropy
from keras.engine.saving import model_from_json
from keras.losses import mse
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from talos.model import lr_normalizer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# GLOBALS
PROJECT_PATH = os.getcwd().replace("/models", "")
BACKUP_PATH = PROJECT_PATH + "/models/backup"
PREDICTIONS_PATH = PROJECT_PATH + "/models/predictions"

# GLOBALS
filename = 'resnet_relu(64)*2_softmax(2)_Adam_lr1.32_bs54.00_meansquarederror_1561576974_45'
params = {
    "losses": mse,
    "optimizer": Adam,
    "lr": 1.32,
}


def imagesToDataset(path):
    data = []

    files = os.listdir(path)
    for filepath in files:
        file = cv2.imread(path + filepath).astype(np.float32)/255
        data.append(file)
    return np.array(data), files


def build_model():
    # load json
    json_file = open(BACKUP_PATH + "/" + filename + '.json', 'r')
    model_json = json_file.read()
    json_file.close()

    # load weights
    new_model = model_from_json(model_json)
    new_model.load_weights(BACKUP_PATH + "/" + filename + ".h5")

    return new_model


dataset, filenames = imagesToDataset(PROJECT_PATH + '/data/test/')


model = build_model()
model.summary()
model.compile(loss=params['losses'],
              optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
              metrics=['accuracy'])

results = model.predict(x=dataset)

# clean predict
hascactus_column_results = np.rint(results[:, [1]]).astype(int)
clean_results = np.column_stack([filenames, hascactus_column_results])

# export
np.savetxt(
    PREDICTIONS_PATH + "/" + filename + ".csv", clean_results,
    header="id,has_cactus",
    delimiter=",", fmt="%s", comments='')
