from datetime import datetime, timedelta

def shift_date(date_str, days=1):
    date_format = "%Y-%m-%d"
    date_obj = datetime.strptime(date_str, date_format)
    new_date_obj = date_obj + timedelta(days=days)
    return new_date_obj.strftime(date_format)


def days_between_dates(date1_str, date2_str):
    date_format = "%Y-%m-%d"
    date1_obj = datetime.strptime(date1_str, date_format)
    date2_obj = datetime.strptime(date2_str, date_format)
    
    delta = date2_obj - date1_obj
    return delta.days

def dataset_time(config,days):
    start_time = config['kwargs']['handler']['kwargs']['start_time']
    end_time = config['kwargs']['handler']['kwargs']['end_time']
    train_time= config['kwargs']['segments']['train']
    valid_time= config['kwargs']['segments']['valid']
    test_time= config['kwargs']['segments']['test']
    config['kwargs']['handler']['kwargs']['start_time']=shift_date(start_time,days)
    config['kwargs']['handler']['kwargs']['end_time']=shift_date(end_time,days)
    config['kwargs']['segments']['train']=[shift_date(date,days) for date in train_time]
    config['kwargs']['segments']['valid']=[shift_date(date,days) for date in valid_time]
    config['kwargs']['segments']['test']=[shift_date(date,days) for date in test_time]
    return config

def port_analysis_config_time(config,days):
    back_start=config['backtest']['start_time']
    back_end=config['backtest']['end_time']
    config['backtest']['start_time']=shift_date(back_start,days)
    config['backtest']['end_time']=shift_date(back_end,days) 
    return config



