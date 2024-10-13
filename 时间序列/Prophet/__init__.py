import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# 模拟生成一些时间序列数据 (例如，销售数据)
np.random.seed(42)
n = 365  # 1年的数据
dates = pd.date_range(start="2023-01-01", periods=n, freq='D')
sales = 50 + np.sin(np.linspace(0, 3*np.pi, n)) * 10 + np.random.normal(0, 5, n)

# 构造DataFrame并重命名列名为Prophet要求的格式
df = pd.DataFrame({'ds': dates, 'y': sales})

# 使用Prophet模型
model = Prophet(daily_seasonality=True)  # 每天的季节性波动
model.fit(df)

# 进行未来30天的预测
future = model.make_future_dataframe(periods=30)  # 生成未来的时间框架
forecast = model.predict(future)  # 进行预测

# 可视化预测结果
fig = model.plot(forecast)
plt.show()

# 可视化组件分析（趋势，季节性等）
fig2 = model.plot_components(forecast)
plt.show()

# 自定义节假日
holidays = pd.DataFrame({
  'holiday': 'holiday_name',
  'ds': pd.to_datetime(['2023-12-25', '2024-01-01']),  # 节假日日期
  'lower_window': 0,
  'upper_window': 1,
})

# 创建带有节假日的Prophet模型
model = Prophet(holidays=holidays)
model.fit(df)

# ---------------------------
# 公式详解：

# 1. **Prophet的主要特点**:
#    - **简单易用**：用户只需要指定日期和目标值，其他的模型选择和参数调优可以自动完成。
#    - **适应趋势变化**：可以处理随时间变化的非线性趋势。
#    - **处理季节性**：能够自动捕捉日、周、年等周期性波动。
#    - **节假日效应**：支持自定义节假日，并能够在预测中考虑这些影响。
#    - **对缺失值和异常值的鲁棒性**：Prophet对缺失值和异常值具有较好的容忍性，适合现实中的不规则数据。

# 2. **Prophet模型的基本原理**:
#    Prophet模型将时间序列建模分为三个主要组成部分：
#    - **趋势 (Trend)**：描述时间序列的长期变化。它可以是线性或非线性（例如S型趋势）。
#    - **季节性 (Seasonality)**：时间序列中的周期性波动，例如日、周、年等。
#    - **节假日效应 (Holiday Effects)**：在特定的节假日或事件期间时间序列的变化。
#
#    Prophet模型的数学公式如下：
#
#    y(t) = g(t) + s(t) + h(t) + ε_t
#    其中：
#    - y(t)：时间点t的观测值。
#    - g(t)：趋势函数，描述时间序列的长期变化。
#    - s(t)：季节性函数，描述周期性波动。
#    - h(t)：节假日效应，描述节假日对时间序列的影响。
#    - ε_t：误差项（噪声），通常假设为独立同分布的误差。

# 3. **Prophet的趋势模型**:
#    - **线性趋势**：趋势是时间的线性函数，适用于没有明显的非线性变化的情况。
#    - **S型趋势**（Logistic Growth）：适用于一些在某一时刻达到饱和或最大值的情况。数学公式为：
#
#    g(t) = C / (1 + exp(-k(t - t_0)))
#    其中：
#    - C：饱和值。
#    - k：增长速率。
#    - t_0：转折点（S型曲线的中点）。
#
#    该模型适用于一些增长速度逐渐减缓、最终趋于饱和的情况。

# 4. **Prophet的季节性**:
#    - 季节性通常是周期性的，如年季节性、周季节性、日季节性。它使用傅里叶级数进行建模，能够捕捉复杂的周期性变化。
#    - 傅里叶级数的表示为：
#
#    s(t) = Σ(A_n * cos(2πn * t / T) + B_n * sin(2πn * t / T))
#    其中：
#    - A_n 和 B_n：傅里叶级数的系数，表示季节性的幅度。
#    - T：季节性周期（例如，日、周、年等）。
#    - t：时间点。
#
#    这使得Prophet能够准确建模复杂的季节性模式，并在未来进行有效的预测。

