import schedule
import time

def update_data_and_model():
    # 将上述代码（从获取数据到保存模型）包装在这个函数中
    pass


# 每天执行一次 update_data_and_model 函数
schedule.every().day.at("00:00").do(update_data_and_model)


# 无限循环，执行定时任务
while True:
    schedule.run_pending()
    time.sleep(1)
