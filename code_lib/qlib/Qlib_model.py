import numpy as np
import pandas as pd
from sklearn import svm
from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.log import get_module_logger
import logging
from qlib.workflow import R
from qlib.utils import flatten_dict
import os
import torch
import matplotlib.pyplot as plt
import yaml
import mlflow
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import json
from qlib.contrib.model.gbdt import LGBModel

from code_lib.base import ML_plt,ML_qlib

# 存储模型
def save_trained_model(model, save_path,model_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_path = os.path.join(save_path, f'{model_name}.bin')
    torch.save(model.model, model_path)
    print(f"Model saved at {model_path}")
    return model_path

# 存储模型配置文件
def save_yaml(experiment,path,name):
    path = path + f"{name}.yaml"
    with open(path, 'w') as yaml_file:
        yaml.dump(experiment, yaml_file)
    pass

# 自定义 SVRModel 类
class SVRModel(Model):
    def __init__(self, **kwargs):
        self.logger = get_module_logger("SVRModel",level = logging.INFO)
        self._params = {}
        self._params.update(kwargs)
        self.model = None
        self.evals_result = []
        
    def fit(self, dataset:DatasetH,evals_result=None):
        data = dataset.prepare("train", col_set=["feature", "label"])
        x_train, y_train = data["feature"], data["label"]

        # 填充NAN值
        x_train = x_train.fillna(0)
        y_train = y_train.fillna(0)

        self.model = svm.SVR(**self._params)
        self.model.fit(x_train, y_train.squeeze())

        # Calculate and save the list of loss values
        y_pred = self.model.predict(x_train)
        self.evals_result = np.mean(np.square(y_pred - y_train.squeeze()))
    
    def predict(self, dataset):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test= dataset.prepare("test", col_set="feature").fillna(0)
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)

    def plot_loss(self):
        # Plot the convergence curve
        plt.plot(self.evals_result)
        plt.xlabel("Iteration")
        plt.ylabel("Loss Value")
        plt.title("SVRModel Training Convergence Curve")
        plt.grid()

        # Save the convergence curve to a specified folder
        convergence_curve_folder = "loss_plt"
        model_name = 'SVRModel'
        os.makedirs(convergence_curve_folder, exist_ok=True)
        plt.savefig(os.path.join(convergence_curve_folder, f"{model_name}.png"))

        # Display the convergence curve
        plt.show()

def model_param(dataset,config,param_path,save_path):
    # 从配置文件中读取参数网格
    with open(param_path, "r") as f:
        param_grid = json.load(f)

    grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error')
    # 切分数据集
    print('切分数据集')
    x_train = dataset.prepare("train", col_set="feature").fillna(0)
    y_train = dataset.prepare("train", col_set="label").fillna(0)
    x_train, y_train = x_train.values, y_train.values.flatten()

    # 进行网格搜索
    print('进行网格搜索')
    grid_search.fit(x_train, y_train)

    # 获取最佳参数
    best_params = grid_search.best_params_

    # 将最佳参数写入新的配置文件
    print('存储最优参数')
    with open(save_path['model'] + "best_params.json", "w") as f:
        json.dump(best_params, f)

    config["model"]["task"]["model"]["kwargs"]= best_params

    print('使用最优参数进行模型训练')
    with R.start(experiment_name="train_model",uri=save_path['model']):
        R.log_params(**flatten_dict(config["model"]["task"]))
        model = ML_qlib.model_init(config["model"])
        evals_result = {}
        model.fit(dataset,evals_result=evals_result)
        R.save_objects(trained_model=model)
        rid = R.get_recorder(experiment_name="train_model").id

    # 预测和评估
    y_pred = model.predict(dataset)
    y_true= dataset.prepare('test').iloc[:, -1:]
    test_loss = mean_squared_error(y_true, y_pred)
    print('打印模型训练loss')
    print("Test loss: ", test_loss)

    # 将评估指标记录到 MLflow
    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metric("test_loss", test_loss)
    return model,rid

# 模型存储
def save_model(model):

    pass

# 加载model
def load_model(model_path):
    
    pass

# CustomLGBModel
class CustomLGBModel(LGBModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_loss = None

    def fit(self, *args, **kwargs):
        evals_result = {}
        super().fit(*args, **kwargs, evals_result=evals_result)
        self.training_loss = evals_result["train"]["l2"]

    @property
    def booster_(self):
        if self.model is not None:
            return self.model
        else:
            raise AttributeError("The model has not been trained yet.")




