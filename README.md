# MLOps-stock-by-Qlib
MLOps：stock by Qlib
github:https://github.com/qidian1029/MLOps_qlib.git
## 环境需求
### 新建python环境 版本：python==3.8
 使用conda：
   conda create -n py3.8 python=3.8
### 更新pip：pip install --user --upgrade pip

# 方法一：cmd 命令行执行
 ## 到experiment路径下
   cd /experiment
 ## 命令行执行
   python gui.py
## 点击需求update_data、update_data、plot_result、test_model、deploy_model按钮
## 打开相应的yaml配置文件，修改，点击按钮：保存并执行或另存并执行

# 方法二：cmd 命令行执行以下内容
到experiment路径下 
  cd /experiment
  执行下列命令
  ## 更新数据
  修改配置文件update_Alpha158_csi500_till_today.yaml
  ### 执行命令
    python update_data.py update_Alpha158_csi500_till_today.yaml
    
  ## 训练模型
  修改配置文件train_lightgbm_Alpha158_csi500.yaml
  ### 执行命令
    python update_data.py train_lightgbm_Alpha158_csi500.yaml
  
  ## 绘制图像
  修改配置文件plot_lightgbm20230419.yaml
  ### 执行命令
    python plot_result.py plot_lightgbm20230419.yaml
    
  ## 测试模型
  修改配置文件test_lightgbm.yaml
  ### 执行命令
    python test_model.py test_lightgbm.yaml
    
  ## 持续部署
  修改配置文件deploy_lightgbm.yaml
  ### 执行命令
    python deploy_model.py deploy_lightgbm.yaml
