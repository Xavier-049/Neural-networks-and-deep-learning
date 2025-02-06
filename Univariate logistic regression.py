import tensorflow as tf  # 导入 TensorFlow 库
print("Tensorflow version: ", tf.__version__)  # 打印 TensorFlow 的版本号

import numpy as np  # 导入 NumPy 库
import matplotlib.pyplot as plt  # 导入 Matplotlib 库用于绘图

# 数据
x = np.array([137.97, 104.50, 100.00, 126.32, 79.20, 99.00, 124.00, 114.00, 106.69, 140.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21])  # 输入特征数据
y = np.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])  # 标签数据（二分类）

# 数据预处理
x_train = x - np.mean(x)  # 将输入特征数据归一化，减去均值
y_train = y  # 标签数据保持不变

# 绘制原始数据
plt.scatter(x_train, y_train)  # 绘制散点图，显示输入特征和标签之间的关系

# 学习率和迭代次数
learn_rate = 0.005  # 设置学习率
iter = 5  # 设置训练迭代次数
display_step = 1  # 每隔多少次迭代打印一次训练信息

# 初始化权重和偏置
np.random.seed(612)  # 设置随机种子，确保结果可复现
w = tf.Variable(np.random.randn())  # 初始化权重，随机生成一个值
b = tf.Variable(np.random.randn())  # 初始化偏置，随机生成一个值

# 绘制初始的 Sigmoid 曲线
x_ = np.linspace(-80, 80, num=100)  # 生成 100 个点，用于绘制 Sigmoid 曲线
y_ = 1 / (1 + tf.exp(-(w * x_ + b)))  # 计算初始 Sigmoid 曲线的值
plt.plot(x_, y_, color='red', linewidth=3)  # 绘制初始 Sigmoid 曲线

# 训练过程
cross_train = []  # 用于存储训练过程中的交叉熵损失
acc_train = []  # 用于存储训练过程中的准确率

for i in range(0, iter + 1):  # 遍历迭代次数
    with tf.GradientTape() as tape:  # 使用 GradientTape 记录梯度信息
        pred_train = 1 / (1 + tf.exp(-(w * x_train + b)))  # 计算预测值，使用 Sigmoid 函数
        Loss_train = -tf.reduce_mean(y_train * tf.math.log(pred_train) + (1 - y_train) * tf.math.log(1 - pred_train))  # 计算交叉熵损失
        Accuracy_train = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred_train < 0.5, 0, 1), y_train), tf.float32))  # 计算准确率

    cross_train.append(Loss_train)  # 将当前损失值添加到列表中
    acc_train.append(Accuracy_train)  # 将当前准确率添加到列表中

    dL_dw, dL_db = tape.gradient(Loss_train, [w, b])  # 计算损失对权重和偏置的梯度

    w.assign_sub(learn_rate * dL_dw)  # 更新权重
    b.assign_sub(learn_rate * dL_db)  # 更新偏置

    if i % display_step == 0:  # 每隔 display_step 次迭代打印一次训练信息
        print("i: %i, Train Loss: %f, Train Accuracy: %f" % (i, Loss_train, Accuracy_train))
        y_ = 1 / (1 + tf.exp(-(w * x_ + b)))  # 使用更新后的权重和偏置重新计算 Sigmoid 曲线
        plt.plot(x_, y_)  # 绘制更新后的 Sigmoid 曲线

plt.show()  # 显示最终的图像