from qlib.data import D
import numpy as np
import torch
import pandas as pd
import pickle
import os
import inspect
from qlib.tests.data import GetData

def preprocess_data(df):
    # 在这个函数中添加数据预处理代码
    pass

def get_stock_data_and_save_as_csv(config):

    path  = os.path.join(config["experiment"]["path"], config["folders"]["data"], config["data"]["input"])
    df = D.features([config["data"]["symbol"]], ['$close'], config["data"]["start_date"], config["data"]["end_date"])
    df.to_csv(path)

def open_stock_data(config):

    experiment_path = config["experiment"]["path"]
    df = pd.read_csv(os.path.join(
    experiment_path, 
    config["folders"]["data"], 
    config["data"]["input"]), 
    index_col=0)

    df = df.rename(columns={"$close": "close"})  
    return df

def data_pre(df_data,config):

    # 创建两个列表，用来存储数据的特征和标签
    data_feat, data_target = [],[]
    # 设每条数据序列有20组数据
    seq = 10
    feature_num = config["model"]["lstm"]["feature_num"]

    for index in range(len(df_data) - seq):
        # 构建特征集
        data_feat.append(df_data['feature'].iloc[:,range(feature_num)][index: index + seq].values)
        # 构建target集
        data_target.append(df_data['label'][index:index + seq])
    # 将特征集和标签集整理成numpy数组
    data_feat = np.array(data_feat)
    data_target = np.array(data_target)
    # 这里按照8:2的比例划分训练集和测试集
    test_set_size = int(np.round(0.1*df_data.shape[0]))  # np.round(1)是四舍五入，
    train_size = data_feat.shape[0] - (test_set_size) 
    trainX = torch.from_numpy(data_feat[:train_size].reshape(-1,seq,feature_num)).type(torch.Tensor)   
    # 这里第一个维度自动确定，我们认为其为batch_size，因为在LSTM类的定义中，设置了batch_first=True
    testX  = torch.from_numpy(data_feat[train_size:].reshape(-1,seq,feature_num)).type(torch.Tensor)
    trainY = torch.from_numpy(data_target[:train_size].reshape(-1,seq,1)).type(torch.Tensor)
    testY  = torch.from_numpy(data_target[train_size:].reshape(-1,seq,1)).type(torch.Tensor)
    
    return trainX,trainY,testX,testY

def data_loader(df_data,config):
    trainX,trainY,testX,testY = data_pre(df_data,config)
    batch_size=config["model"]["batch_size"]
    train = torch.utils.data.TensorDataset(trainX,trainY)
    test = torch.utils.data.TensorDataset(testX,testY)

    train_loader = torch.utils.data.DataLoader(dataset=train, 
                                           batch_size=batch_size, 
                                           shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test, 
                                          batch_size=batch_size, 
                                          shuffle=False)
    
def data_get_csv(df):

    pass

def tensor_cat_dataframe(tensor1, tensor2,columns_list):

    # 将两个张量拼接在一起（沿第1维）
    concatenated_tensor = torch.cat((tensor1, tensor2), dim=1)

        # 将拼接后的张量转换为 numpy 数组
    concatenated_array = concatenated_tensor.numpy()

    # 将 numpy 数组转换为 pandas DataFrame
    df = pd.DataFrame(concatenated_array, columns=columns_list)

    print(df.head())
    return df

# 将元组保存到文件中
def save_tuple_to_file(data_tuple, file_path):
    """
    Save a tuple to a specified file path using the pickle module.

    Args:
        data_tuple (tuple): A tuple containing data to be saved.
        file_path (str): The file path where the tuple will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as f:
        pickle.dump(data_tuple, f)

#文件中读取元组
def load_tuple_from_file(file_path):
    """
    Load a tuple from a specified file path using the pickle module.

    Args:
        file_path (str): The file path where the tuple is stored.

    Returns:
        tuple: The loaded tuple.
    """
    with open(file_path, 'rb') as f:
        data_tuple = pickle.load(f)
    return data_tuple

def dataframe_cat():

    pass
    
def save_dataframe_to_csv(df, folder_path, filename):
    """
    Save a DataFrame as a CSV file to a specified folder path.

    Args:
        df (pd.DataFrame): The DataFrame to save as a CSV file.
        folder_path (str): The path to the folder where the CSV file will be saved.
        filename (str): The name of the CSV file.
    """
    # 构建 CSV 文件的完整路径
    file_path = os.path.join(folder_path, filename)
    
    # 保存 DataFrame 为 CSV 文件
    df.to_csv(file_path, index=False)

def get_variable_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def save_df_to_csv(df,folder_path,exp_name):
    # 保存 DataFrame 为 CSV 文件
    df.to_csv(folder_path, f"{exp_name}.csv",index=True,)

# 设置二级列名，转换表格数据，适配qlib的dataset数据输入要求
def transform_dataframe(df, n_feature_columns):
    # df数据表，n_feature_columns特征数
    # 设置 datetime 和 instrument 为行索引
    # df = df.set_index(['datetime', 'instrument'])

    # 自动将前 n_feature_columns 列设为 feature，剩余列设为 label
    new_columns = [('feature', col) if i < n_feature_columns else ('label', col)
                   for i, col in enumerate(df.columns)]

    df.columns = pd.MultiIndex.from_tuples(new_columns)
    
    return df

# 通过qlib自带GetData()更新数据
def qlib_upgrade_data(target_dir):
    GetData().qlib_data(target_dir=target_dir,exists_skip=False,delete_old=True)
    # 解压压缩包并删除久的压缩包
    import os
    import zipfile
    from datetime import date
    from zipfile import BadZipFile

    # 获取更新当日日期并构造文件名前缀
    today = date.today()
    prefix = today.strftime('%Y%m%d')
        
    # 遍历指定文件夹内的所有.zip文件，如果文件名以前缀开头，则解压缩文件到当前文件夹
    folder = target_dir
    for file in os.listdir(folder):
        print(file)
        if file.endswith('.zip') and file.startswith(prefix):
            try:
                with zipfile.ZipFile(os.path.join(folder, file), 'r') as zip_ref:
                    zip_ref.extractall(folder)
            except BadZipFile:
                print(f"Error: {file} is not a valid zip file.")
        if file.endswith('.zip') and not file.startswith(prefix):
            os.remove(os.path.join(folder, file))
    print("数据更新成功")
    pass


