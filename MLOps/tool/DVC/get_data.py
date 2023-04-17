import qlib
from qlib.data import D



# 初始化 Qlib
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')


# 获取原始数据
data = D.features(["SH600000"], ['$close', '$volume'], start_time='2010-01-01', end_time='2020-12-31')


# 保存原始数据到 CSV 文件
data.to_csv("raw_data.csv")
