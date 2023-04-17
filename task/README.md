修改完配置文件，在项目根目录文件夹执行python task/main.py
# use case_1
修改配置文件
把训练时间设置为2016.9-2020.9
dataset:
    kwargs:  
        handler:  
            kwargs:
                start_time: '2016-07-01' #dataset开始时间
                end_time: '2020-08-31' # dataset结束时间
        segments: # 设置train、valid、test的时间范围
            test: ['2020-07-01', '2020-08-31']
回测设置为2017.7-2017.8
dataset:
    kwargs:  
        segments: # 设置train、valid、test的时间范围
            test: ['2020-07-01', '2020-08-31']
port_analysis_config:
    backtest: 
        start_time: '2020-07-01' # 回测开始时间
        end_time: '2020-08-31' # 回测结束时间
对比绘图，保存实验结果
compare: # 对比图像开关
    report_normal_df: True # 是否生成对比交易图
    analysis_df: False # 是否结果对比图
# use case_2
设置超参数空间，修改model参数
修改param_grid.json文件，修改配置文件
  - train_name: model_2
    experiment:
        name: model_2
        path: ./ 
    config:
        optimize_hyperparams: True # 是否进行超参数优化
        param_grid_path : ./param_grid.json # 参数空间位置
        loss: False
        accuracy: False
    model: 
        task: 
            model:
                class: SVRModel # 自定义模型类名
                module_path: code_lib.Qlib_model #自定义模型类位置
                kwargs:
# use case_3
打开update_data开关，修改end_day参数到2023.02.01
continuous_deployment:
    update_data: True # 是否更新当前下载的存储数据
    end_day: '2023-02-01' # 更新回测截止时间
# use case_4
配置continuous参数，False为单次执行，后续为模型制定数据参数，和模型的位置和配置参数，修改dataset.yaml的数据test时间范围
continuous_deployment:
    continuous: False # 是否进行持续部署，自动每天执行一次
    dataset:
        config: ./task_4/data/dataset.yaml 
    model:
        config: ./task_4/model/model_1.yaml 
        path:  ./task_4/model/model_1.bin 
# use case_5
未训练模型时，加载模型，配置模型类名，类位置，超参数设置
experiments_list: # 实验模型配置
  - train_name: model_2
    experiment:
        name: model_2
        path: ./ 
    config:
        optimize_hyperparams: True # 是否进行超参数优化
        param_grid_path : ./param_grid.json # 参数空间位置
        loss: False
        accuracy: False
    model: 
        task: 
            model:
                class: SVRModel # 自定义模型类名
                module_path: code_lib.Qlib_model #自定义模型类位置
                kwargs:

已训练模型时，修改以下参数，获得回测结果
continuous_deployment:
    model:
        config: ./task_4/model/model_1.yaml 
        path:  ./task_4/model/model_1.bin 





