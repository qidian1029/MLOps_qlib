# main.py
import yaml
import sys
import os

def main():
    current_working_directory = os.getcwd()
    # 获取上一级文件夹路径
    parent_folder_path = os.path.dirname(current_working_directory)
    sys.path.append(parent_folder_path)
    from code_lib import ML_qlib,ML_file,Process_design,Qlib_model

    # 加载配置文件
    with open("config.yaml", "r",encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 创建文件夹
    ML_file.create_file(config)

    # 设置mlflow存储位置
    mlflow_storage_location = os.path.dirname(config["folders"]["data"])
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlflow_storage_location}"

    # 初始化qlib
    ML_qlib.qlib_init(config,mlflow_storage_location)

    # 准备dataset
    Qlib_model.save_yaml(config['dataset'],config["folders"]["data"],'dataset')
    dataset = Process_design.prepar_dataset(config)

    # 模型训练
    for i, experiment in enumerate(config["experiments_list"]):
        print(f"Running experiment {i + 1}...")
        if config["process_design"]["train"] == True:
            model,rid = Process_design.train_model(dataset,experiment,config["folders"])
            Qlib_model.save_trained_model(model,config["folders"]["model"],experiment['train_name'])
            Qlib_model.save_yaml(experiment['model']['task'],config["folders"]["model"],experiment['train_name'])
            Process_design.qlib_backtest(config,dataset,model,rid,experiment)

    # 回测对比图
    Process_design.compare(config)

    # 持续部署
    if config['process_design']['continuous']==True:
        Process_design.continuous(config)
    
    pass

if __name__ == "__main__":
    main()