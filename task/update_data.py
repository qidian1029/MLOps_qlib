import yaml
import sys
import os
import argparse
import certifi
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

def update_data(config_file):
    current_working_directory = os.getcwd()
    parent_folder_path = os.path.dirname(current_working_directory)
    sys.path.append(parent_folder_path)
    from code_lib import ML_data

    # Load configuration file
    from yamlinclude import YamlIncludeConstructor
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.SafeLoader, base_dir='.')
    from code_lib import config_set
    base_config_path = "base_setting.yaml"
    config = config_set.merge_configs(base_config_path, config_file)
    if config['update_data']['tool']['qlib']==True:
        ML_data.qlib_upgrade_data(config['qlib']['uri'])
        print('更新完成')
    else:
        print('不需要更新qlib_data') 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with the specified configuration file.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    update_data(args.config_file)

