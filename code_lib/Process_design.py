import datetime
import os
import yaml
import torch
import pandas as pd

# qlib包
from qlib.utils import init_instance_by_config,flatten_dict
from qlib.workflow import R
from qlib.data.dataset.handler import DataHandler
from qlib.data.dataset.loader import StaticDataLoader
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.contrib.evaluate import risk_analysis 
from qlib.backtest import backtest as qlib_back

# 自建
from code_lib import Qlib_model,ML_qlib,Qlib_dataset,Process_design,config_updata_time,ML_data,ML_plt

# 1.数据准备，输出dataset
def data_to_dataset(config):
    data_path = None
    if config["dataset"]['path'] is not None:
        data_path.update(config["dataset"]['path'])
    if data_path ==None:
        dataset = init_instance_by_config(config["dataset"])
    else:
        data = pd.read_csv(data_path,index_col= [0,1],header=[0,1],parse_dates=True).fillna(0)
        data.index.set_names(["datetime", "instrument"], inplace=True)
        static_data_loader = StaticDataLoader(config = data, join = "outer")
        data = static_data_loader.load()
        handler = DataHandler(data_loader=static_data_loader)   
        config["dataset"]["kwargs"]["handler"] = handler
        dataset = init_instance_by_config(config["dataset"])
    return dataset

def prepar_dataset(config):
    if config["process_design"]["dataset"] == True:

        data_path = config["folders"]["data"] + "dataset.csv"

        if config["dataset"]["path"] is not None:
            data_path = config["dataset"]["path"]
            dataset_1 = Qlib_dataset.data_to_dataset(config,data_path)
        else:
            # 模型及数据初始化
            dataset = Process_design.data_to_dataset(config)

            # 存储dataset
            Qlib_dataset.qlib_dataset_save(dataset,config)

            # 读取dataset
            dataset_1 = Qlib_dataset.data_to_dataset(config,data_path)
    return dataset_1


# 2.模型训练
def train_model(dataset,config,path):
    default_config = {
        "loss": False,
        "optimize_hyperparams": False,
        "accuracy": False,
    }
    if config["config"] is not None:
        default_config.update(config["config"])
  
    if default_config["optimize_hyperparams"]:
        model,rid = Qlib_model.model_param(dataset,config,default_config["param_grid_path"],path)
    else:
        with R.start(experiment_name="train_model",uri=path['model']):
            R.log_params(**flatten_dict(config["model"]["task"]))
            model = ML_qlib.model_init(config["model"])
            if config['config']['loss']==True:
                evals_result = {}
                model.fit(dataset,evals_result = evals_result)
                ML_plt.qlib_plt_loss(evals_result["train"]["l2"],config['train_name'],path['plt'])
            else:
                model.fit(dataset)
            R.save_objects(trained_model=model)
            rid = R.get_recorder().id

    from sklearn.metrics import mean_squared_error, accuracy_score
    train_dataloader = dataset.prepare("train")
    valid_dataloader = dataset.prepare("valid")

    if default_config["accuracy"]:
        train_pred = model.predict(train_dataloader)
        valid_pred = model.predict(valid_dataloader)
        train_accuracy = accuracy_score(train_dataloader.get_label(), train_pred)
        valid_accuracy = accuracy_score(valid_dataloader.get_label(), valid_pred)
        print(f"Train accuracy: {train_accuracy:.4f}, Validation accuracy: {valid_accuracy:.4f}")

    if default_config["loss"]:
        y_pred = model.predict(dataset)
        y_true= dataset.prepare('test').iloc[:, -1:]
        test_loss = mean_squared_error(y_true, y_pred)
        print("Test loss: ", test_loss)
    
    return model,rid


# 3.回测预部署
def qlib_backtest(config,dataset,model,rid,experiment):
    config["port_analysis_config"]["strategy"]["kwargs"]["model"]= model
    config["port_analysis_config"]["strategy"]["kwargs"]["dataset"]= dataset
    port_analysis_config = config["port_analysis_config"]
    # backtest and analysis
    with R.start(experiment_name="backtest_analysis"):
        recorder = R.get_recorder(recorder_id=rid, experiment_name="train_model")
        model = recorder.load_object("trained_model")

        # prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # backtest & analysis
        par = PortAnaRecord(recorder,port_analysis_config, "day")
        par.generate()

        pred_df = recorder.load_object("pred.pkl")
        report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl") 
        analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    csv_name = experiment["train_name"]
    result_path = config["folders"]["result"]

    pred_df.to_csv(result_path+'/pred_df/'+f"{csv_name}.csv")
    report_normal_df.to_csv(result_path+'/report_normal_df/'+f"{csv_name}.csv")
    analysis_df.to_csv(result_path+'/analysis_df/'+f"{csv_name}.csv")

    return pred_df,report_normal_df,analysis_df


# 回测对比绘图
def compare(config):
    compare_config_result = {
        "report_normal_df": False,
        "analysis_df": False,
    }
    result_path = config["folders"]["result"]
    if config["compare"] is not None:
        compare_config_result.update(config["compare"])

    if compare_config_result["report_normal_df"]:
        ML_plt.compare_report_normal_df(result_path + "report_normal_df/")

    if compare_config_result["analysis_df"]:
        ML_plt.compare_analysis_df(result_path + "analysis_df/")


# 4.持续部署
# 更新数据图表
def update_csv_df(report_normal,config):
    parent_folder = config["folders"]["continuous"] 
    import datetime
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    # Combine the parent folder path with the new folder name
    new_folder_path = os.path.join(parent_folder, current_date)
    # Create the new folder if it doesn't exist
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    ML_plt.report_plt(report_normal,new_folder_path+'/')

    analysis = dict()
    analysis["excess_return_without_cost"] = risk_analysis(
                        report_normal["return"] - report_normal["bench"], freq='day'
                    )
    analysis["excess_return_with_cost"] = risk_analysis(
                        report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq='day'
                    )
    analysis_df = pd.concat(analysis)  # type: pd.DataFrame
    analysis_file_path = config["folders"]["continuous"]  + "continuous_analysis_df.csv"
    if not os.path.exists(analysis_file_path):
        analysis_df.columns = [f"{current_date}_{col}" for col in analysis_df.columns]
        analysis_df.to_csv(analysis_file_path)
    else:
        # Read the existing CSV file
        existing_df = pd.read_csv(analysis_file_path, index_col=(0,1))
        # Rename the DataFrame columns with the current date
        analysis_df.columns = [f"{current_date}_{col}" for col in analysis_df.columns]
        # Concatenate the existing DataFrame and the new DataFrame
        updated_df = pd.concat([existing_df, analysis_df], axis=1)
        # Save the updated DataFrame to the CSV file
        updated_df.to_csv(analysis_file_path)
    
    pass


# 持续部署
def continuous_deployment(config):
    update_data = config['continuous_deployment']['update_data']
    continuous = config['continuous_deployment']['continuous']
    end_day = config['continuous_deployment']['end_day']
    if continuous == True:
        current_date = datetime.now()
        date_string = current_date.strftime("%Y-%m-%d")
        end_day = date_string

    ML_data.qlib_upgrade_data(update_data)

    # Instantiate the dataset using the configuration
    with open(config['continuous_deployment']['dataset']['config'], "r",encoding="utf-8") as f:
        dataset_config = yaml.safe_load(f)
    
    days = config_updata_time.days_between_dates(dataset_config['kwargs']['handler']['kwargs']['end_time'],end_day)
    dataset_config = config_updata_time.dataset_time(dataset_config,days)
    dataset = init_instance_by_config(dataset_config)

    # Load the trained model from the specified path
    model_path = config['continuous_deployment']['model']['path']
    with open(config['continuous_deployment']['model']['config'], "r",encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
    print(model_config)
    model = init_instance_by_config(model_config['model'])
    model.model = torch.load(model_path)
    # Instantiate the strategy using the configuration
    strategy_config = config["port_analysis_config"]["strategy"]
    strategy_config['kwargs']["model"] = model
    strategy_config['kwargs']['dataset'] = dataset

    strategy = init_instance_by_config(strategy_config)

    # Instantiate the executor using the configuration
    executor_config = config["port_analysis_config"]["executor"]
    executor = init_instance_by_config(executor_config)

    # Perform backtesting using the configuration
    config["port_analysis_config"] = config_updata_time.port_analysis_config_time(config["port_analysis_config"],days)
    
    backtest_start_time = config["port_analysis_config"]["backtest"]["start_time"]
    backtest_end_time = config["port_analysis_config"]["backtest"]["end_time"]
    benchmark = config["port_analysis_config"]["backtest"]["benchmark"]
    account = config["port_analysis_config"]["backtest"]["account"]
    exchange_kwargs = config["port_analysis_config"]["backtest"]["exchange_kwargs"]
    pos_type = "Position"

    portfolio_metric_dict, indicator_dict = qlib_back(
        start_time=backtest_start_time,
        end_time=backtest_end_time,
        strategy=strategy,
        executor=executor,
        benchmark=benchmark,
        account=account,
        exchange_kwargs=exchange_kwargs,
        pos_type=pos_type,
    )
    from qlib.utils.resam import Freq
    freq = exchange_kwargs['freq']
    analysis_freq = "{0}{1}".format(*Freq.parse(freq))
    report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)
    # 持续更新回测结果图表
    update_csv_df(report_normal,config)

    pass


#每天执行
import schedule
import time
from datetime import datetime

def print_date():
    current_date = datetime.now()
    date_string = current_date.strftime("%Y-%m-%d")
    print(date_string)


def continuous(config):
    if config['continuous_deployment']['continuous']==True:
        schedule.every().day.at("00:00").do(print_date)
        while True:
            continuous_deployment(config)
            schedule.run_pending()
            time.sleep(60)
    else:
        continuous_deployment(config)
        
    pass
