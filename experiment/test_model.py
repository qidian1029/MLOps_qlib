import yaml
import sys
import os
import argparse
from code_lib import Process_design
from code_lib.base import ML_qlib, ML_file,config_set
from yamlinclude import YamlIncludeConstructor

def test_model(config_file):
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

    folder_path = ML_file.create_folder(config["experiment"]["path"],'deployment')

    # 持续部署
    if config['process_design']['deployment']==True:
        Process_design.deploy(config,folder_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with the specified configuration file.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    test_model(args.config_file)
