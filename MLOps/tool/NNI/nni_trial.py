import nni
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from code_lib import ML_qlib,ML_file
import yaml

def init():
    with open("qlib_train.yaml", "r",encoding="utf-8") as f:
        config = yaml.safe_load(f)
    ML_qlib.qlib_init(config)
    # 创建文件夹
    ML_file.create_file(config)
    # 模型及数据初始化
    dataset = ML_qlib.dataset_init(config)
    return dataset

def get_data(dataset):
    x_test= dataset.prepare("test",  col_set="feature").fillna(0)
    y_test = dataset.prepare("test", col_set="label").fillna(0)
    x_train = dataset.prepare("train", col_set="feature").fillna(0)
    y_train = dataset.prepare("train", col_set="label").fillna(0)
    x_train, y_train = x_train.values, y_train.values.flatten()
    return x_train, y_train, x_test, y_test

def main(params):
    x_train, y_train, x_test, y_test = get_data(init())
    # 训练模型
    model = SVR(C=params['C'], epsilon=params['epsilon'], kernel=params['kernel'])
    model.fit(x_train.values, y_train.values.flatten())
    # 预测
    y_pred = model.predict(x_test.values)
    # 计算误差
    mse = mean_squared_error(y_test, y_pred)
    # 报告结果
    nni.report_final_result(mse)

if __name__ == '__main__':
    try:
        # 获取超参数
        tuned_params = nni.get_next_parameter()
        print("Tuned parameters:", tuned_params)
        main(tuned_params)
    except Exception as e:
        print("Exception:", str(e))
        raise
