# Machine Learning : cactus identification
Identify cactus with aerial images of desert

Competition here : https://www.kaggle.com/c/aerial-cactus-identification/



### todo
- [x] MLP model
- [x] ResNet model
- [x] Hyperparameters setup
- [x] Predictions setup
- [x] <a href="https://github.com/autonomio/talos">Talos</a> for random search and grid search
- [x] <a href="https://mlflow.org/">MLFlow</a> implementation for logs and benchmark analysis



### logs with Tensorboard
```Bash
tensorboard --logdir=logs_tensorboard &
```

### logs with MLFlow
```Bash
mlflow server --host 0.0.0.0 --backend-store-uri file:///project_path/logs_mlflow
```

