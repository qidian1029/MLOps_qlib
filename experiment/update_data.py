import yaml
import sys
import os
import certifi
from yamlinclude import YamlIncludeConstructor

os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

def update_data():
    current_working_directory = os.getcwd()
    parent_folder_path = os.path.dirname(current_working_directory)
    sys.path.append(parent_folder_path)

    # Load configuration file
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.SafeLoader, base_dir='.')

    base_config_path = "base_setting.yaml"
    from code_lib.base import ML_data

    with open(base_config_path, "r", encoding="utf-8") as task_file:
        config = yaml.safe_load(task_file)
    market_path = './data/cn_data/instruments/stock_dates.txt'
    market_path = os.path.join(parent_folder_path,market_path)
    collector_path = os.path.join(parent_folder_path,config['qlib']['collector_path'])
    dump_bin_path = os.path.join(parent_folder_path,config['qlib']['dump_bin_path'])
    data_path = os.path.join(parent_folder_path,config['qlib']['uri'])
    
    config['qlib']['collector_path'] = collector_path
    config['qlib']['dump_bin_path'] = dump_bin_path
    config['qlib']['data_path'] = data_path
    config['qlib']['stock_txt_path'] = market_path

    from code_lib.qlib import update_data

    # 指定你想要的目录
    directory = data_path+'/instruments'
    # 检查目录是否存在
    if not os.path.exists(directory):
    # 如果不存在，创建它
        os.makedirs(directory)

    ML_data.akshare_upgrade_data(data_path=directory)
    update_data.ak_data_to_csv(config)

if __name__ == "__main__":
    update_data()

