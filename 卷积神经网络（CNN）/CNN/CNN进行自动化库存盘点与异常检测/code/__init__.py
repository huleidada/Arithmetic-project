import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# 生成模拟数据
# 假设我们生成 (1000, 64, 64, 3) 的图像数据集，1000张商品图像，64x64像素，3通道（RGB）
# 这些数据代表仓库货架上的商品图像，包含正常和异常情况（如错位、缺货或损坏）
X_data = np.random.rand(1000, 64, 64, 3)  # 商品图像（正常和异常的模拟）
y_data = np.random.randint(2, size=(1000, 1))  # 标签：0表示正常，1表示异常（例如：错位、缺货或损坏）

# 拆分为训练集和测试集
# 将数据集拆分为80%的训练集和20%的测试集，用于模型训练和评估
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# 数据增强：通过旋转、平移、剪切等增强训练数据集的多样性，提升模型的鲁棒性。
# 这样模型可以适应不同的商品摆放角度和环境变化（例如：货架上光照变化或角度不同）
datagen = ImageDataGenerator(
    rotation_range=20,  # 随机旋转范围为20度
    width_shift_range=0.2,  # 水平平移
    height_shift_range=0.2,  # 垂直平移
    shear_range=0.2,  # 剪切变换
    zoom_range=0.2,  # 缩放范围
    horizontal_flip=True,  # 随机水平翻转
    fill_mode="nearest"  # 填充模式
)

# 适用数据增强于训练集，提升模型的泛化能力
datagen.fit(X_train)

# 构建卷积神经网络模型
model = models.Sequential()

# 卷积层1：卷积操作是通过一个卷积核（滤波器）对图像进行局部特征提取。
# 公式： 输出特征图 = Conv(W * Input + b)
# 其中 W 是卷积核，b 是偏置
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))  # 输入64x64x3的RGB图像
model.add(layers.MaxPooling2D((2, 2)))  # 池化层：通过池化操作将特征图缩小，减少计算量

# 卷积层2：提取更复杂的特征
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# 卷积层3：继续增加卷积核的深度和复杂度，逐步提取更高级的特征
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# 扁平化层：将多维的特征图转换为一维向量，便于全连接层处理
model.add(layers.Flatten())  # 公式： Flatten(输出特征图) -> 一维向量

# 全连接层1：通过全连接层将提取的特征与神经元连接，进行分类决策。
# 公式： Dense(输出) = Activation(W * Input + b)
model.add(layers.Dense(128, activation='relu'))

# 输出层：二分类任务，使用sigmoid激活函数输出是否异常（0或1）
# 公式： Sigmoid(x) = 1 / (1 + exp(-x))
model.add(layers.Dense(1, activation='sigmoid'))

# 编译模型
# 使用Adam优化器和二分类交叉熵损失函数
# 公式： BinaryCrossentropy = -[y * log(p) + (1 - y) * log(1 - p)]
# 其中 y 是实际标签，p 是预测概率
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 使用数据增强训练模型
# 通过扩充数据集来避免模型过拟合，提升模型对不同商品的识别能力
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(X_test, y_test)
)

# 评估模型在测试集上的表现
# 测试集用于验证模型在未见过的数据上的泛化能力
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

# 绘制训练过程中的准确率和损失图
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# 模拟预测一张新的商品图像（假设new_image是一个64x64的RGB图像）
# 通过模型对实时拍摄的商品图像进行分类，判断其是否存在异常
new_image = np.random.rand(1, 64, 64, 3)

# 预测新图像是否异常
prediction = model.predict(new_image)
print(f'Predicted probability of anomaly: {prediction[0][0]}')

if prediction >= 0.5:
    print("Predicted: Anomaly (e.g., misplaced or damaged)")  # 如果预测值大于0.5，预测为异常
else:
    print("Predicted: Normal")  # 如果预测值小于0.5，预测为正常
