import pandas as pd
from qlib.data.dataset.handler import DataHandler
from qlib.data.dataset.loader import StaticDataLoader
from qlib.utils import init_instance_by_config
from code_lib.base import ML_data

def data_to_dataset(config,data_path):
    data = pd.read_csv(data_path,index_col= [0,1],header=[0,1],parse_dates=True).fillna(0)
    data.index.set_names(["datetime", "instrument"], inplace=True)
    static_data_loader = StaticDataLoader(config = data, join = "outer")
    data = static_data_loader.load()
    handler = DataHandler(data_loader=static_data_loader)   
    config["dataset"]["kwargs"]["handler"] = handler
    dataset = init_instance_by_config(config["dataset"])
    return dataset

def qlib_dataset_save(dataset,config):
    num = len(dataset.prepare("train","feature").columns.to_list())#num为label的个数

    train_df = ML_data.transform_dataframe(dataset.prepare("train"),num)
    valid_df = ML_data.transform_dataframe(dataset.prepare("valid"),num)
    test_df = ML_data.transform_dataframe(dataset.prepare("test"),num)
    
    # 存储数据
    dataset = pd.concat([train_df,valid_df,test_df])
    dataset.to_csv(config["folders"]["data"]+"dataset.csv") 
    train_df.to_csv(config["folders"]["data"]+"train_dataset.csv")
    valid_df.to_csv(config["folders"]["data"]+"valid_dataset.csv")
    test_df.to_csv(config["folders"]["data"]+"test_dataset.csv")
    pass

