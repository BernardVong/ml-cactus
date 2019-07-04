import os
import time

import cv2
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model
from keras.activations import relu, softmax, tanh, sigmoid
from keras.callbacks import TensorBoard, LambdaCallback
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation, \
    AveragePooling2D, Add, Dropout
from keras.losses import mse, binary_crossentropy
from keras.optimizers import sgd, Adam
from keras.regularizers import l2
from keras.utils import np_utils
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from talos import Scan
from talos.model import lr_normalizer
from talos.utils.gpu_utils import parallel_gpu_jobs


# MacOS : fix compatibility
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# GLOBALS
PROJECT_PATH = os.getcwd().replace("/models", "")

# GLOBALS ENV
ENV = "remote"
MLFLOW_PATH = PROJECT_PATH + '/logs_mlflow' if ENV == "local" else '/mnt/mlflow/logs'
MODELS_BACKUP = PROJECT_PATH + "/models/backup" if ENV == "local" else '/mnt/mlflow/backup'

# GLOBALS MODEL
NUM_CLASSES = 2
PROJECT_MODEL = "mlp"
model = None
MODEL_CURRENT_PARAMS = None


mlf_client = MlflowClient(tracking_uri=MLFLOW_PATH)
mlf_experiment = mlf_client.get_experiment_by_name("cactus")

params = {
    'lr': (0.2, 3, 10),
    'first_neuron': [32, 64],
    'hidden_layers': (4, 8, 10),
    'batch_size': [64, 128],
    'epochs': [25],
    'dropout': (0, 0.5, 5),
    'weight_regulizer': [None],
    'emb_output_dims': [None],
    'shapes': ['brick', 'funnel'],
    'optimizer': [Adam],
    'losses': [mse, binary_crossentropy],
    'activation': [tanh],
    'last_activation': [sigmoid]
}


''' DATA PROCESSING '''
def imagesToDataset(paths):
    data = []
    for filepath in paths:
        file = cv2.imread(filepath).astype(np.float32)/255
        data.append(file)
    return np.array(data)

def data_processing():
    '''
    :return: x_train, x_test, y_train, y_test
    '''

    train_csv = pd.read_csv(PROJECT_PATH + '/data/train.csv')
    files = [PROJECT_PATH + '/data/train/' + file for file in train_csv['id'].tolist()]
    trains = imagesToDataset(files)
    labels = np_utils.to_categorical(train_csv['has_cactus'].tolist(), 2)
    return train_test_split(trains, labels, test_size=0.33)
''' DATA PROCESSING END '''


''' LOGS MANAGERS '''
def logs_name(p):
    current_params = {
        "lr": format(p['lr'], '.2f'),
        "bs": format(p['batch_size'], '.2f'),
        "first_neuron": str(p['first_neuron']),
        "layers": str(p['hidden_layers'] + 1),
        "activation": p['activation'].__name__,
        "last_activation": p['last_activation'].__name__,
        "optimizer": p['optimizer'].__name__,
        "losses": p['losses'].__name__.replace("_", ""),
        "timestamp": str(round(time.time())),
        "logs_name": "",
        "logs_name_path": ""
    }

    logs_model = PROJECT_MODEL
    logs_folder = PROJECT_PATH + "/logs_tensorboard/mlp_trash/"
    logs_layers = current_params["activation"] + "(" + current_params["first_neuron"] + ")*" + current_params["layers"] + "_" + current_params["last_activation"] + "(" + str(NUM_CLASSES) + ")"
    logs_hyperparams = current_params["optimizer"] + "_lr" + current_params["lr"] + "_bs" + current_params["bs"] + "_" + current_params["losses"]
    current_params["logs_name"] = logs_model + "_" + logs_layers + "_" + logs_hyperparams + '_' + current_params["timestamp"]
    current_params["logs_name_path"] = logs_folder + current_params["logs_name"]

    # save global
    global MODEL_CURRENT_PARAMS
    MODEL_CURRENT_PARAMS = current_params

    return current_params["logs_name_path"]

def logs_on_epoch_end(epoch, logs=None):
    if not logs:
        return

    # SETUP PARAMS
    mlf_client.log_param(run_id=mlf_run.info.run_id, key="model", value=MODEL_CURRENT_PARAMS["logs_name"])
    mlf_client.log_param(run_id=mlf_run.info.run_id, key="activation", value=MODEL_CURRENT_PARAMS["activation"])
    mlf_client.log_param(run_id=mlf_run.info.run_id, key="optimizer", value=MODEL_CURRENT_PARAMS["optimizer"])
    mlf_client.log_param(run_id=mlf_run.info.run_id, key="losses", value=MODEL_CURRENT_PARAMS["losses"])

    # SETUP METRICS
    mlf_client.log_metric(run_id=mlf_run.info.run_id, key="lr", value=float(MODEL_CURRENT_PARAMS["lr"]), step=epoch)
    mlf_client.log_metric(run_id=mlf_run.info.run_id, key="bs", value=float(MODEL_CURRENT_PARAMS["bs"]), step=epoch)
    mlf_client.log_metric(run_id=mlf_run.info.run_id, key="first_neuron", value=float(MODEL_CURRENT_PARAMS["first_neuron"]), step=epoch)
    mlf_client.log_metric(run_id=mlf_run.info.run_id, key="layers", value=float(MODEL_CURRENT_PARAMS["layers"]), step=epoch)
    mlf_client.log_metric(run_id=mlf_run.info.run_id, key="timestamp", value=float(MODEL_CURRENT_PARAMS["timestamp"]), step=epoch)

    # SETUP METRICS RESULTS
    mlf_client.log_metric(run_id=mlf_run.info.run_id, key="acc", value=logs["acc"], step=epoch)
    mlf_client.log_metric(run_id=mlf_run.info.run_id, key="loss", value=logs["loss"], step=epoch)
    mlf_client.log_metric(run_id=mlf_run.info.run_id, key="val_acc", value=logs["val_acc"], step=epoch)
    mlf_client.log_metric(run_id=mlf_run.info.run_id, key="val_loss", value=logs["val_loss"], step=epoch)


    model_save_name = MODELS_BACKUP + "/" + MODEL_CURRENT_PARAMS["logs_name"] + "_" + str(epoch)

    # save model JSON
    model_json = model.to_json()
    with open(model_save_name + ".json", "w") as json_file:
        json_file.write(model_json)

    # save model H5
    model.save_weights(model_save_name + ".h5")


def logs_refresh_mlf_run(logs=None):
    global mlf_run
    mlf_run = mlf_client.create_run(mlf_experiment.experiment_id)
''' LOGS MANAGERS END '''


def cactus_model(x_train, y_train, x_test, y_test, p):

    # create and train model
    global model
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32, 3)))

    model.add(Dense(p['first_neuron'], activation=p['activation'], kernel_initializer='normal'))
    model.add(Dropout(p['dropout']))
    hidden_layers(model, p, 1)
    model.add(Dense(NUM_CLASSES, activation=p['last_activation'], kernel_initializer='normal'))

    model.compile(loss=p['losses'],
                  optimizer=p['optimizer'](lr=lr_normalizer(p['lr'], p['optimizer'])),
                  metrics=['accuracy'])
    model.summary()

    # LOGS
    tb_callback = TensorBoard(log_dir=logs_name(p))

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=p['batch_size'],
        epochs=p['epochs'],
        callbacks=[
            tb_callback,
            LambdaCallback(
                on_train_begin=logs_refresh_mlf_run,
                on_epoch_end=logs_on_epoch_end
            )
        ],
        verbose=0
    )


    return history, model


x_train, x_test, y_train, y_test = data_processing()
scanResults = Scan(
    x=x_train, y=y_train,
    x_val=x_test, y_val=y_test,
    model=cactus_model,
    params=params,
    dataset_name="cactus",
    experiment_no=str(round(time.time()))
)
