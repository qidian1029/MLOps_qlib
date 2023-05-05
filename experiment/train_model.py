import yaml
import sys
import os
import argparse
from code_lib import Process_design
from code_lib.base import ML_qlib, ML_file,config_set
from code_lib.qlib import Qlib_model
from yamlinclude import YamlIncludeConstructor

def main(config_file):
    current_working_directory = os.getcwd()
    parent_folder_path = os.path.dirname(current_working_directory)
    sys.path.append(parent_folder_path)

    # Load configuration file
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.SafeLoader, base_dir='.')
    base_config_path = "base_setting.yaml"
    config = config_set.merge_configs(base_config_path, config_file)
    
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
        model,rid = Process_design.train_model(dataset,experiment,config["folders"])
        Qlib_model.save_trained_model(model,config["folders"]["model"],experiment['train_name'])
        Qlib_model.save_yaml(experiment['model']['task'],config["folders"]["model"],experiment['train_name'])
        Process_design.qlib_backtest(config,dataset,model,rid,experiment)
    
    # ... rest of the code ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with the specified configuration file.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config_file)
