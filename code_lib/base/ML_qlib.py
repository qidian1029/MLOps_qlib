import qlib
import pandas as pd
from qlib.workflow.expm import MLflowExpManager
from qlib.config import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
from qlib.contrib.report import analysis_model, analysis_position
from qlib.data import D
import sys
import os
from code_lib.base import ML_data

def qlib_init(config,path):
    print("qlib初始化开始")
    current_working_directory = os.getcwd()
    # 获取上一级文件夹路径
    parent_folder_path = os.path.dirname(current_working_directory)
    sys.path.append(parent_folder_path)
    config["qlib"]["uri"] = os.path.join(parent_folder_path,config["qlib"]["uri"])
    print('qlib_uri:',config["qlib"]["uri"])
    qlib.init(provider_uri=config["qlib"]["uri"],mount_path=path,region=REG_CN)
    print("qlib初始化完成")
    return 

def model_init(config):
    print('模型初始化开始')
    model = init_instance_by_config(config["task"]["model"])
    print('模型初始化完成')
    return model

def dataset_init(config):
    dataset = init_instance_by_config(config["task"]["dataset"])
    return dataset

def model_train(model,dataset,config):
    exp_manager = MLflowExpManager(uri=config["folders"]["model"])# 修改mlflow文件存储位置
    
    # start exp to train model
    with R.start(experiment_name="train_model"):
        R.log_params(**flatten_dict(config["task"]))
        model.fit(dataset)
        R.save_objects(trained_model=model)
        rid = R.get_recorder().id

    config["port_analysis_config"]["strategy"]["kwargs"]["model"]= model
    config["port_analysis_config"]["strategy"]["kwargs"]["dataset"]= dataset
    port_analysis_config = config["port_analysis_config"]
    # backtest and analysis
    with R.start(experiment_name="backtest_analysis"):
        recorder = R.get_recorder(recorder_id=rid, experiment_name="train_model")
        model = recorder.load_object("trained_model")

        # prediction
        recorder = R.get_recorder()
        # ba_rid = recorder.id
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # backtest & analysis
        par = PortAnaRecord(recorder,port_analysis_config, "day")
        par.generate()

    pred_df = recorder.load_object("pred.pkl")
    # pred_df_dates = pred_df.index.get_level_values(level='datetime')
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    #positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    result_path = config["folders"]["result"]
    ML_data.save_df_to_csv(pred_df,result_path)
    ML_data.save_df_to_csv(report_normal_df,result_path)
    ML_data.save_df_to_csv(analysis_df,result_path)

    return model,pred_df,report_normal_df,analysis_df

