import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import mplfinance as mpf
import fbprophet

# 从文件中获取数据
df = pd.read_excel('test1.xls')
df.info()

#s数据清洗：检查数据是否有空值
df.isnull().values.sum()

#调整数据索引为data
df.set_index(["date"], inplace=True)
df.index = pd.to_datetime(df.index)

df = df[['open', 'close', 'high', 'low', 'volume']]
# featureData = df[['open', 'high', 'volume', 'low']]

#计算股票涨跌幅
df['Range'] = df['close'] - df['close'].shift(1)

#简单查看股票价格走势图
df['close'].plot(figsize = (20,4),
                     color = 'r',
                     alpha = 0.8,
                     grid = True,
                     rot = 0,
                     title = '简单查看股票走势')


#绘制K线图可以使用mplfinance绘制

mpf.plot(df, type='candle',mav=(3,6,9), volume=True)

#绘制数据标准化后的组合图Min—Max标准化
df_min_max = (df - df.min())/(df.max()-df.min())
df_min_max .plot(figsize = (16,9))

#使用 Pandas提供的rolling函数计算均线
y = df['2020-06-16':'2021-07-21']
y_close = y.Close
short_rolling = y_close.rolling(window=5).mean()
long_rolling = y_close.rolling(window=15).mean()


fig, ax = plt.subplots(figsize=(16,4))
ax.plot(short_rolling.index, short_rolling, label='5 days rolling')
ax.plot(long_rolling.index, long_rolling, label='15 days rolling')
ax.set_xlabel('Date')
ax.set_ylabel('Closing price (¥)')
ax.legend(fontsize='large')
#侯建模型
fig, ax = plt.subplots(figsize=(16,9))
short_long = np.sign(short_rolling - long_rolling)
buy_sell = np.sign(short_long - short_long.shift(1))
buy_sell.plot(ax=ax)
ax.axhline(y=0, color='red', lw=2)

data = y['close'].reset_index()
data = data.rename(columns={'date': 'ds', 'close': 'y'})

#数据建模
model = fbprophet.Prophet(changepoint_prior_scale=0.05, daily_seasonality=True) # 定义模型
model.fit(data) # 训练模型
forecast_df = model.make_future_dataframe(periods=365, freq='D') # 生成需预测序列
forecast = model.predict(forecast_df) # 模型预测
model.plot(forecast, xlabel = 'Date', ylabel = 'Close Price ¥') # 绘制预测图
plt.title('Close Price of 600036.SS')


