#!/usr/bin/env bash

# CREATE ENV
# mlflow experiments create --experiment-name cactus

# local
mlflow ui --backend-store-uri file:///Users/bernard.vong/projects/ml-cactus_identification/logs_mlflow

# remote
