import yaml
from yamlinclude import YamlIncludeConstructor

def merge_configs(base_config_path, task_config_path):
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.SafeLoader, base_dir='.')

    with open(task_config_path, "r", encoding="utf-8") as task_file:
        task_config = yaml.safe_load(task_file)

    merged_config = {**task_config["common"], **task_config}
    return merged_config
