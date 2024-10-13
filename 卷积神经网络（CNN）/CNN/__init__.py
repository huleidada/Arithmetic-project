import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 假设我们生成一个大小为 (100, 64, 64, 3) 的随机图像数据集
# 100是样本数量，64x64 是图像尺寸，3表示RGB通道
X_train = np.random.rand(100, 64, 64, 3)  # 随机生成的图像数据
Y_train = np.random.randint(2, size=(100, 1))  # 随机生成的标签，0或1，表示正常或异常

# CNN模型定义
model = models.Sequential()

# 卷积层1：提取图像中的局部特征
# 卷积操作公式：
# f(x, y) = (I * K)(x, y) = ∑m ∑n I(m, n) · K(x-m, y-n)
# 其中，I 是输入图像，K 是卷积核，(x, y)表示输出特征图的坐标
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# 卷积层2：进一步提取高级特征
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# 卷积层3：更深层的特征提取
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# 扁平化层：将卷积后的特征图展平
# 扁平化公式：将二维特征图展平为一维向量
# Flatten：f(x1, x2, x3, ...) -> [x1, x2, x3, ...]
model.add(layers.Flatten())

# 全连接层：用于分类
# 全连接公式：y = W · x + b
# 其中，W 是权重矩阵，x 是输入特征向量，b 是偏置项，y 是输出
model.add(layers.Dense(128, activation='relu'))

# 输出层：使用Sigmoid激活函数进行二分类
# Sigmoid函数公式：σ(z) = 1 / (1 + e^(-z))
# Sigmoid将输出值限制在0到1之间，适用于二分类问题
model.add(layers.Dense(1, activation='sigmoid'))

# 编译模型
# 使用binary_crossentropy损失函数和adam优化器
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, Y_train, epochs=10, validation_split=0.2, batch_size=32)

# 绘制训练过程中的准确率和损失图
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# 代码注释中的公式解释：

# 1. 卷积操作：
#    卷积层用于提取图像中的局部特征，卷积操作的数学公式如下：
#    f(x, y) = (I * K)(x, y) = ∑m ∑n I(m, n) · K(x-m, y-n)
#    其中，I 是输入图像，K 是卷积核，(x, y)表示输出特征图的坐标。
#    卷积操作是通过滑动卷积核在输入图像上进行逐元素相乘并求和，得到特征图。

# 2. 池化操作：
#    池化层（如最大池化）用于减少特征图的尺寸，同时保留最重要的特征。
#    池化操作的公式如下：
#    MaxPooling(x, y) = max{Patch(x, y)}
#    其中，Patch(x, y) 代表池化窗口中的像素值，最大池化选择该窗口的最大值。

# 3. 扁平化层：
#    扁平化操作的目的是将卷积层提取的二维特征图展平为一维向量。
#    扁平化操作的公式如下：
#    Flatten：f(x1, x2, x3, ...) -> [x1, x2, x3, ...]
#    这里的输出是一个一维向量，作为全连接层的输入。

# 4. 全连接层：
#    全连接层的输入是经过卷积和池化层提取出的特征，输出是用于分类的结果。
#    全连接层的数学公式如下：
#    y = W · x + b
#    其中，W 是权重矩阵，x 是输入特征向量，b 是偏置项，y 是输出。

# 5. 输出层（Sigmoid）：
#    在二分类问题中，输出层通常使用 Sigmoid 激活函数来将输出值限制在0和1之间，表示类别概率。
#    Sigmoid函数的公式如下：
#    σ(z) = 1 / (1 + e^(-z))
#    其中，z 是全连接层的线性组合 W · x + b。
