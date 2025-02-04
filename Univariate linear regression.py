# 导入必要的库
import numpy as np  # 用于数值计算
import tensorflow as tf  # 用于深度学习和张量操作
import matplotlib.pyplot as plt  # 用于绘图

# 设置matplotlib的字体为黑体，以便中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei']

# 定义输入数据x，表示房屋面积（单位：平方米）
x = tf.constant([[137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00,
                  106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21]])

# 定义目标数据y，表示房屋价格（单位：万元）
y = tf.constant([[145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00,
                  62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69, 95.30]])

# 计算x的均值
meanX = tf.reduce_mean(x)

# 计算y的均值
meanY = tf.reduce_mean(y)

# 计算分子：(x - meanX) * (y - meanY) 的总和
sumXY = tf.reduce_sum((x - meanX) * (y - meanY))

# 计算分母：(x - meanX) * (x - meanX) 的总和
sumX = tf.reduce_sum((x - meanX) * (x - meanX))

# 计算线性回归的斜率w
w = sumXY / sumX

# 计算线性回归的截距b
b = meanY - w * meanX

# 打印计算得到的权值w和偏置b
print("权值w=", w.numpy(), "\n偏置值b=", b.numpy())
print("线性模型:y=", w.numpy(), "*x+", b.numpy())

# 定义测试数据x_test，表示需要预测的房屋面积
x_test = np.array([128.15, 45.00, 141.43, 106.27, 99.00, 53.84, 85.36, 70.00])

# 使用线性模型计算预测的房价y_pred
y_pred = (w * x_test + b).numpy()

# 打印测试数据及其对应的预测房价
print("面积\t估计房价")
n = len(x_test)
for i in range(n):
    print(x_test[i], "\t", round(y_pred[i], 2))

# 创建一个新的图形窗口
plt.figure()

# 绘制原始数据点（红色散点图），表示销售记录
plt.scatter(x, y, color="red", label="销售记录")

# 绘制预测数据点（蓝色散点图），表示预测房价
plt.scatter(x_test, y_pred, color="blue", label="预测房价")

# 绘制拟合直线（绿色线），表示线性回归模型
plt.plot(x_test, y_pred, color="green", label="拟合直线")

# 设置x轴标签
plt.xlabel("面积（平方米）", fontsize=14)

# 设置y轴标签
plt.ylabel("价格（万元）", fontsize=14)

# 设置x轴的范围
plt.xlim((40, 150))

# 设置y轴的范围
plt.ylim((40, 150))

# 设置图形的标题
plt.suptitle("商品房销售价格评估系统v1.0", fontsize=20)

# 添加图例，位置在左上角
plt.legend(loc="upper left")

# 显示图形
plt.show()
