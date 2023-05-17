import akshare as ak
import pandas as pd
import os

def update_market(config):
    collector_path = config['qlib']['collector_path']
    data_path = config['update_data']['data_path']
    import subprocess
    market = ['CSI100','CSI300','CSI500']
    for m in market:
        command = rf"python {collector_path} --index_name " + f"{m} --qlib_dir "+f"{data_path} --method parse_instruments"
        print(command)
        result = subprocess.run(command, shell=True, text=True, capture_output=True)

        print("命令行输出：")
        print(result.stdout)
        print("命令行错误：")
        print(result.stderr)
    return print("市场更新完成")

def csv_bin(dump_path,csv_path,qlib_dir):
    print("将csv转化为bin格式数据")
    import subprocess
    command = f"python {dump_path} dump_all --csv_path "+f"{csv_path} --qlib_dir {qlib_dir} --symbol_field_name stock_code --date_field_name date --include_fields open,high,low,close,volume,money,factor,vwap,change"

    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    print(command)
    print("命令行输出：")
    print(result.stdout)
    print("命令行错误：")
    print(result.stderr)

    print("bin格式数据已存储到:",qlib_dir)
    return print()

# akshare更新数据库
def ak_data_to_csv(config):
    dump_path = config['qlib']['dump_bin_path']
    path = config['qlib']['stock_txt_path']
    data_path = config['qlib']['data_path']
    csv_path = f'{data_path}/stock'
    with open(path, "r") as f:
        lines = f.readlines()
        stock_info = [line.strip().split("\t") for line in lines]
    os.makedirs(csv_path, exist_ok=True)
    for stock_code, start_date, end_date in stock_info:
        start_date = start_date.replace('-', '')
        current_date = end_date
        end_date = end_date.replace('-', '')

        stock_code = stock_code.lower()
        df = ak.stock_zh_a_hist(symbol=stock_code[2:],period="daily", start_date=start_date, end_date=end_date)
        print(df)
        adj_factor = ak.stock_zh_a_daily(symbol=stock_code, start_date=start_date, end_date=end_date, adjust="qfq-factor")
        df = df.drop(columns=['振幅', '涨跌幅','涨跌额','换手率'])
        df = df.rename(columns={
            '日期': 'date', 
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'money',     
            })

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)


        # 获取第一行的 qfq_factor 值
        first_row_qfq_factor = adj_factor.loc[0, 'qfq_factor']
        # 在第一行插入新的数据
        adj_factor.loc[-1] = [current_date, first_row_qfq_factor]  # 这是临时的行，我们将它放在 DataFrame 的开始
        adj_factor.index = adj_factor.index + 1  # 把所有的索引都向下移动一个位置
        adj_factor = adj_factor.sort_index()  # 对索引进行排序，这将使得新插入的行成为第一行
        
        adj_factor["date"] = pd.to_datetime(adj_factor["date"])
        adj_factor.set_index("date", inplace=True)
        adj_factor.sort_index(inplace=True)

        # Generate a date range from the minimum to maximum date
        date_range = pd.date_range(adj_factor.index.min(), adj_factor.index.max(), freq='D')

        # Reindex the DataFrame with the generated date range
        adj_factor = adj_factor.reindex(date_range)

        # Forward fill the missing values in 'qfq_factor' column
        adj_factor['qfq_factor'] = adj_factor['qfq_factor'].ffill()

        # Reset the index and rename it to 'date'
        adj_factor = adj_factor.reset_index().rename(columns={'index': 'date'})

        data = pd.merge(df, adj_factor, on=["date"])
        data.set_index("date", inplace=True)
        data = data.apply(pd.to_numeric, errors='coerce')
        data['stock_code'] = stock_code
        data["adj_close"] = data["close"] * data["qfq_factor"]
        data["adj_open"] = data["open"] * data["qfq_factor"]
        data["adj_high"] = data["high"] * data["qfq_factor"]
        data["adj_low"] = data["low"] * data["qfq_factor"]
        data["vwap"] = (data["money"] / data["volume"]).round(2)
        data["change"] = data["adj_close"].pct_change()
        data = data[["stock_code","adj_open", "adj_high", "adj_low", "adj_close", "volume", "money", "vwap", "change","qfq_factor"]]
        data.columns = ["stock_code","open", "high", "low", "close", "volume", "money", "vwap", "change","factor"]
        # 存储文件
        data.to_csv(f'{csv_path}/{stock_code.lower()}.csv', index=True)
        print(data)
        print("已更新：",stock_code)

    csv_bin(dump_path,csv_path,data_path)
    return print("数据更新完成")


