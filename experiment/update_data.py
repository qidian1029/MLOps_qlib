import yaml
import sys
import os
import argparse
import certifi
from yamlinclude import YamlIncludeConstructor

os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

def update_data(config_file):
    current_working_directory = os.getcwd()
    parent_folder_path = os.path.dirname(current_working_directory)
    sys.path.append(parent_folder_path)

    # Load configuration file
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.SafeLoader, base_dir='.')

    base_config_path = "base_setting.yaml"
    from code_lib.base import ML_data,config_set
    
    config = config_set.merge_configs(base_config_path, config_file)
    if config['update_data']['tool']['qlib']==True:
        path = os.path.join(parent_folder_path,config['qlib']['uri'])
        ML_data.qlib_upgrade_data(path)
        print('更新完成,cn_data存储地址：',path)
    else:
        print('不需要更新qlib_data') 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with the specified configuration file.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    update_data(args.config_file)

