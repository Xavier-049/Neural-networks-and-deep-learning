import tensorflow as tf
print("Tensorflow version: ", tf.__version__)

import numpy as np
import matplotlib.pyplot as plt

# 数据
x = np.array([137.97, 104.50, 100.00, 126.32, 79.20, 99.00, 124.00, 114.00, 106.69, 140.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21])
y = np.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])

# 数据预处理
x_train = x - np.mean(x)
y_train = y

# 绘制原始数据
plt.scatter(x_train, y_train)

# 学习率和迭代次数
learn_rate = 0.005
iter = 5
display_step = 1

# 初始化权重和偏置
np.random.seed(612)
w = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

# 绘制初始的 Sigmoid 曲线
x_ = np.linspace(-80, 80, num=100)  # 生成 100 个点
y_ = 1 / (1 + tf.exp(-(w * x_ + b)))
plt.plot(x_, y_, color='red', linewidth=3)

# 训练过程
cross_train = []
acc_train = []

for i in range(0, iter + 1):

    with tf.GradientTape() as tape:
        pred_train = 1 / (1 + tf.exp(-(w * x_train + b)))
        Loss_train = -tf.reduce_mean(y_train * tf.math.log(pred_train) + (1 - y_train) * tf.math.log(1 - pred_train))
        Accuracy_train = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred_train < 0.5, 0, 1), y_train), tf.float32))

    cross_train.append(Loss_train)
    acc_train.append(Accuracy_train)

    dL_dw, dL_db = tape.gradient(Loss_train, [w, b])

    w.assign_sub(learn_rate * dL_dw)
    b.assign_sub(learn_rate * dL_db)

    if i % display_step == 0:
        print("i: %i, Train Loss: %f, Train Accuracy: %f" % (i, Loss_train, Accuracy_train))
        y_ = 1 / (1 + tf.exp(-(w * x_ + b)))  # 使用 x_ 来计算 y_
        plt.plot(x_, y_)

plt.show()