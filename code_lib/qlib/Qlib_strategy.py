from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

# 继承内置选股策略
class CustomTopkDropoutStrategy(TopkDropoutStrategy):
    def __init__(self, *args, **kwargs):
        super(CustomTopkDropoutStrategy, self).__init__(*args, **kwargs)

    # 在这里添加或覆盖任何您需要修改的方法


import numpy as np
import pandas as pd
from qlib.strategy.base import BaseStrategy

# 自定义选股策略
class BestReturnStrategy(BaseStrategy):
    def __init__(self, lookback_period, top_k=5, *args, **kwargs):
        super(BestReturnStrategy, self).__init__(*args, **kwargs)
        self.lookback_period = lookback_period
        self.top_k = top_k

    def generate_order_list(self, instrument, end_date=None):
        close_prices = self.get_data(instrument, "close", end_date)
        returns = close_prices.pct_change(self.lookback_period)
        best_stocks = returns.apply(lambda x: x.nlargest(self.top_k).index, axis=1).dropna()

        order_list = []
        for date, stocks in best_stocks.iteritems():
            for stock in stocks:
                order_list.append(
                    {
                        "date": date,
                        "instrument": stock,
                        "amount": 1,  # 持有 1 份最佳表现的股票
                        "direction": "buy",  # 买入
                    }
                )
        return order_list

    def get_data(self, instrument, field, end_date):
        start_date = self.start_time
        if end_date is not None:
            end_date = pd.Timestamp(end_date) - pd.Timedelta(days=1)
        return self.provider.get_data(instrument, field, start_date, end_date)


import pandas as pd
import talib as ta

class MovingAverageStrategy(BaseStrategy):
    def __init__(self, win_short, win_long):
        self.win_short = win_short
        self.win_long = win_long

    def strategy_ma(self, pdatas):
        datas = pdatas.copy()
        datas['lma'] = ta.MA(datas['close'], timeperiod=self.win_long)
        datas['sma'] = ta.MA(datas['close'], timeperiod=self.win_short)
        # ... (rest of the strategy code)

    def generate_alpha(self, instrument=None):
        # Get the stock data
        stock_data = self.get_data(instrument)

        # Apply the moving average strategy
        strategy_results = self.strategy_ma(stock_data)

        # Generate the alpha series
        alpha_series = pd.Series(index=strategy_results.index, data=0)

        long_signals = strategy_results['flag'] == 1
        short_signals = strategy_results['flag'] == -1

        alpha_series[long_signals] = 1
        alpha_series[short_signals] = -1

        return alpha_series
