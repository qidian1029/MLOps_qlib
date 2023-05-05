import yaml
import sys
import yaml
import sys
import os
import argparse
from code_lib import Process_design
from code_lib.base import config_set

from yamlinclude import YamlIncludeConstructor

def plot_result(config_file):
    current_working_directory = os.getcwd()
    parent_folder_path = os.path.dirname(current_working_directory)
    sys.path.append(parent_folder_path)

    # Load configuration file
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.SafeLoader, base_dir='.')
    base_config_path = "base_setting.yaml"
    config = config_set.merge_configs(base_config_path, config_file)

    # 回测对比图
    Process_design.compare(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with the specified configuration file.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    plot_result(args.config_file)

