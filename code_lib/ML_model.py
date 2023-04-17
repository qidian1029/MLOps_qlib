import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from qlib.contrib.model.pytorch_lstm import LSTMModel
from qlib.utils import init_instance_by_config
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional	
import seaborn as sns

def model_to_file(model,path):
    # 保存模型
    torch.save(model.state_dict(), path+'model.pth')


def model_get_file(model):
    model_path = 'lstm_model.pth'
    model_config = {
    "input_size": 4,
    "hidden_size": 32,
    "output_size": 1,
    "num_layers": 1
    }
    model.load_state_dict(torch.load(model_path))
    pass


class LSTM(nn.Module):
    """
        使用LSTM进行回归
        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x


def run_lstm(df_data,config):

    from code_lib import ML_data
    feature_num = config["model"]["lstm"]["feature_num"]
    trainX,trainY,testX,testY = ML_data.data_pre(df_data,config)
    ML_data.data_loader(df_data,config)

    input_dim = feature_num    # 数据的特征数
    hidden_dim = config["model"]["lstm"]["hidden_dim"]    # 隐藏层的神经元个数
    num_layers = config["model"]["lstm"]["num_layers"]    # LSTM的层数
    output_dim = config["model"]["lstm"]["output_dim"]     # 预测值的特征数
                     #（这是预测股票价格，所以这里特征数是1，如果预测一个单词，那么这里是one-hot向量的编码长度）

    # 定义模型
    num_epochs = config["train"]["epochs"]
    model = LSTM(input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim, num_layers=num_layers)
    #   打印模型各层的参数尺寸
    # for i in range(len(list(model.parameters()))):
    #     print(list(model.parameters())[i].size())
    
    loss_function = nn.MSELoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # 优化器
    
    train_loss = [] 
    for epoch in range(num_epochs):
        out = model(trainX)
        loss = loss_function(out, trainY)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(loss.item())
        #if (epoch + 1) % 2 == 0:
        #   print('Epoch: {}, Loss:{:.10f}'.format(epoch + 1, loss.item()))

    from code_lib import ML_plt
    ML_plt.loss_plt(train_loss,config["folders"]["plt"])
    

    #模型存储
    model_to_file(model,config["folders"]["model"])
    return model

def qlib_lstm_load():
    from qlib.contrib.model.gbdt import LGBModel
    config = {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        }
    model = LGBModel(**config) # model
    return model


