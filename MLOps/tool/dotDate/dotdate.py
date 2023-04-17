from dotdate.flow import Flow
from dotdate.targets.dvc import DvcTarget

with Flow("my_flow") as flow:
    # 添加数据依赖
    data = flow.add("stock_data.csv.dd")

    # 添加预处理任务
    preprocess = flow.add("preprocess.py", inputs=[data])

    # 添加训练任务
    train = flow.add("train.py", inputs=[preprocess])

    # 添加评估任务
    evaluate = flow.add("evaluate.py", inputs=[train])

    # 输出结果
    flow.add_output(DvcTarget("results"))
