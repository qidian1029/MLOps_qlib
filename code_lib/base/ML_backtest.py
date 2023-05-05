import numpy as np
import pandas as pd
import qlib
from qlib.contrib.model.pytorch_lstm import LSTMModel
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import (
    backtest as compute_backtest,
    get_perf,
)
from qlib.utils import init_instance_by_config
from contextlib import closing

def run_backtest(model_path, data_path, strategy_config, backtest_config):
    """
    使用 Qlib 的回测框架对已保存的 LSTM 模型进行回测。
    :param model_path: 已保存 LSTM 模型的路径。
    :param data_path: 输入数据的路径。
    :param strategy_config: 策略配置字典。
    :param backtest_config: 回测评估器配置字典。
    
    """
    # 加载已保存的 LSTM 模型
    lstm_model = LSTMModel.load(model_path)

    # 为策略配置设置模型
    strategy_config["kwargs"]["model"] = lstm_model

    # 使用 Qlib 的回测框架进行回测
    with closing(compute_backtest(**backtest_config)) as (runner, _):
        strategy = init_instance_by_config(strategy_config)
        runner.run(strategy)
