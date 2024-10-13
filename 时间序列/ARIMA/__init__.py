import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# 加载数据
data = pd.read_csv('time_series_data.csv')
time_series = data['value']

# 进行ADF检验检查平稳性
result = adfuller(time_series)
print(f'p-value: {result[1]}')

# 如果p-value > 0.05，进行差分以使时间序列平稳
diff_series = time_series.diff().dropna()

# 绘制ACF和PACF图，确定p和q
plot_acf(diff_series)
plot_pacf(diff_series)
plt.show()

# 拟合ARIMA模型（假设p=1, d=1, q=1）
model = ARIMA(time_series, order=(1, 1, 1))
fit_model = model.fit()

# 查看模型摘要
print(fit_model.summary())

# 进行预测
forecast = fit_model.forecast(steps=12)
print(forecast)
