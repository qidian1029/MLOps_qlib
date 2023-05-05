# MLOps-stock-by-Qlib
MLOps：stock by Qlib
github:https://github.com/qidian1029/MLOps_qlib.git


# cmd 命令行执行以下内容
到experiment路径下 
  cd /experiment
  执行下列命令
  ## 更新数据
   修改配置文件
    python update_data.py update_Alpha158_csi500_till_today.yaml
    
  ## 训练模型
  修改配置文件
    python train_model.py train_lightgbm_Alpha158_csi500.yaml
  
  ## 绘制图像
  修改配置文件
    python plot_result.py plot_lightgbm20230419.yaml
    
  ## 测试模型
  修改配置文件
    python test_model.py test_lightgbm.yaml
    
  ## 持续部署
  修改配置文件
    python deploy_model.py deploy_lightgbm.yaml
