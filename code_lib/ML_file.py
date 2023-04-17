import os
def create_folder(parent_directory, folder_name):
    new_folder_path = os.path.join(parent_directory, folder_name)

    # 如果文件夹不存在，创建它
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)


def create_file(config):

    # 获取实验名称和路径
    experiment_name = config["experiment"]["name"]
    experiment_path = os.path.join(config["experiment"]["path"], experiment_name)

    folders = config["folders"]

    # 创建实验文件夹及子文件夹
    for folder in folders:
        path = os.path.join(experiment_path, folder)
        os.makedirs(path, exist_ok=True)   
        config["folders"][folder] = config["experiment"]["path"]+config["experiment"]["name"] + '/'+folder+'/'
    
    result_path = config["folders"]['result']
    create_folder(result_path,'pred_df')
    create_folder(result_path,'report_normal_df')
    create_folder(result_path ,'analysis_df')
    return config


