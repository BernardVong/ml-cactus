#!/usr/bin/env bash

# CREATE ENV
# mlflow experiments create --experiment-name default
# mlflow experiments create --experiment-name cactus

# local
export MLFLOW_TRACKING_URI=/Users/bernard.vong/projects/ml-cactus_identification/logs_mlflow
mlflow ui --backend-store-uri file:///Users/bernard.vong/projects/ml-cactus_identification/logs_mlflow

# remote
export MLFLOW_TRACKING_URI=/home/bernard.vong/cactus/logs_mlflow
mlflow server --host 0.0.0.0 --backend-store-uri file:///home/bernard.vong/cactus/logs_mlflow

# remote with Firestore
#sudo apt-get -y update
sudo apt-get install -y nfs-common
sudo mkdir /mnt/mlflow
sudo mnt 10.65.107.10:/logs /mnt/mlflow
sudo chmod go+rwx /mnt/mlflow/
mkdir /mnt/mlflow/logs
mkdir /mnt/mlflow/backup
mkdir ~/cactus

export MLFLOW_TRACKING_URI=/mnt/mlflow/logs
mlflow server --host 0.0.0.0 --backend-store-uri file:///mnt/mlflow/logs
