import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 生成一个模拟时间序列数据，包含趋势、季节性和噪声
# 真实场景下，应该加载历史天气或销售数据
np.random.seed(42)
n = 120  # 数据点数量，代表10年的数据（月度）
dates = pd.date_range(start="2010-01-01", periods=n, freq='M')  # 生成时间序列的日期（每月）
trend = 0.05 * np.arange(n)  # 线性趋势
seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 12)  # 季节性成分（12个月周期）
noise = np.random.normal(0, 2, n)  # 随机噪声
time_series = trend + seasonal + noise  # 合成时间序列数据

# 创建DataFrame，将日期作为索引
df = pd.DataFrame({'Date': dates, 'value': time_series})
df.set_index('Date', inplace=True)  # 将日期列设置为索引

# 检查时间序列是否平稳
# ADF检验（Augmented Dickey-Fuller Test）用于检测序列的平稳性
result = adfuller(df['value'])
print(f'ADF检验的p-value: {result[1]}')  # p-value < 0.05说明时间序列是平稳的

# 如果p-value > 0.05，数据是非平稳的，需要进行差分
# 进行季节性差分以消除季节性趋势
df['diff'] = df['value'].diff().dropna()  # 计算一阶差分

# 绘制ACF和PACF图，确定非季节性部分的p、q和季节性部分P、Q
# ACF (Auto-Correlation Function)：自相关图，用于识别自回归部分p和季节性自回归部分P
# PACF (Partial Auto-Correlation Function)：偏自相关图，用于识别移动平均部分q和季节性移动平均部分Q
plot_acf(df['diff'].dropna())  # 自相关图
plot_pacf(df['diff'].dropna())  # 偏自相关图
plt.show()

# SARIMA模型的参数
# SARIMA模型的阶数：p, d, q分别对应ARIMA部分的自回归、差分和移动平均
# P, D, Q分别是季节性部分的自回归、差分和移动平均
# s是季节周期，这里假设为12（月度数据的季节性周期为12）
p, d, q = 1, 1, 1  # ARIMA部分的参数
P, D, Q, s = 1, 1, 1, 12  # SARIMA部分的参数，季节周期s=12（适用于月度数据）

# 使用SARIMA模型拟合时间序列数据
# SARIMAX是SARIMA模型的实现，'order'为非季节性部分的(p,d,q)，'seasonal_order'为季节性部分的(P,D,Q,s)
model = SARIMAX(df['value'], order=(p, d, q), seasonal_order=(P, D, Q, s))
fit_model = model.fit()  # 拟合模型

# 输出模型拟合结果的摘要
print(fit_model.summary())  # 查看模型摘要，了解系数、显著性等信息

# 进行未来12个月的预测
# 使用get_forecast进行未来时间步的预测，steps=12表示预测未来12个月的数据
forecast = fit_model.get_forecast(steps=12)
forecast_mean = forecast.predicted_mean  # 预测的均值
conf_int = forecast.conf_int()  # 预测的置信区间

# 可视化结果
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['value'], label='历史数据')  # 绘制历史数据
plt.plot(pd.date_range(start=df.index[-1], periods=13, freq='M')[1:], forecast_mean, label='预测', color='red')  # 绘制预测数据
plt.fill_between(pd.date_range(start=df.index[-1], periods=13, freq='M')[1:], conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)  # 绘制置信区间
plt.legend()
plt.title('SARIMA模型预测')
plt.show()

# ---------------------------
# 公式详解：

# 1. 时间序列生成：
#    - 趋势：线性趋势生成公式：
#    trend = 0.05 * t
#    其中，t 是时间点。
#
#    - 季节性：正弦波形模拟季节性波动公式：
#    seasonal = 10 * sin(2 * pi * t / 12)
#    这表示每12个月重复一次的季节性波动。
#
#    - 随机噪声：随机噪声采用正态分布生成：
#    noise ~ N(0, 2)
#    其中，均值为0，标准差为2。

# 2. 平稳性检验 (ADF检验)：
#    - ADF检验（Augmented Dickey-Fuller Test）用于检查时间序列是否平稳。平稳性是建模时的重要假设。ADF检验的原假设是“时间序列存在单位根，即非平稳”。
#    - ADF检验的数学假设：
#      - 原假设 H_0：数据存在单位根，即非平稳。
#      - 备择假设 H_1：数据不含单位根，即平稳。
#    - 如果 p-value 小于 0.05，说明序列平稳。

# 3. 差分：
#    - 如果时间序列不是平稳的，可以通过差分来使其平稳。差分公式为：
#    Z_t = Y_t - Y_{t-1}
#    其中，Z_t 表示差分后的时间序列，Y_t 和 Y_{t-1} 分别是当前时刻和前一时刻的观察值。

# 4. ACF 和 PACF 图：
#    - ACF (Auto-Correlation Function)：自相关图，用于识别ARIMA模型的自回归部分 p 和季节性自回归部分 P。
#    - PACF (Partial Auto-Correlation Function)：偏自相关图，用于识别移动平均部分 q 和季节性移动平均部分 Q。
#
#    - 自回归 (AR) 模型：用过去的观察值预测当前值，公式为：
#    Y_t = phi_1 * Y_{t-1} + phi_2 * Y_{t-2} + ... + phi_p * Y_{t-p} + epsilon_t
#    其中，phi_1, phi_2, ..., phi_p 为自回归系数，epsilon_t 为误差项。
#
#    - 移动平均 (MA) 模型：用过去的误差项来预测当前值，公式为：
#    Y_t = mu + epsilon_t + theta_1 * epsilon_{t-1} + theta_2 * epsilon_{t-2} + ... + theta_q * epsilon_{t-q}
#    其中，theta_1, theta_2, ..., theta_q 为移动平均系数。

# 5. SARIMA 模型：
#    - SARIMA 模型扩展了 ARIMA 模型，加入了季节性成分。其数学表达式为：
#    (1 - phi_1 * B - phi_2 * B^2 - ... - phi_P * B^P)(1 - B^s)^{D} * Y_t = epsilon_t + theta_1 * B + theta_2 * B^2 + ... + theta_Q * B^Q + theta_1 * B^s + ... + theta_Q * B^{Qs}
#    其中，B 是滞后算子，s 是季节性周期，phi 为自回归系数，theta 为移动平均系数，D 为季节性差分阶数。

# 6. 预测公式：
#    - 通过拟合的 SARIMA 模型，我们可以进行未来的预测。假设 Y_t 是当前时刻的观测值，未来时刻的预测值 Y_hat_{t+h} 由以下公式给出：
#    Y_hat_{t+h} = mu + phi_1 * Y_{t+h-1} + ... + phi_P * Y_{t+h-P} + theta_1 * epsilon_{t+h-1} + ... + theta_Q * epsilon_{t+h-Q}
#    其中，mu 是常数项，epsilon 是误差项，h 是预测的步数。

