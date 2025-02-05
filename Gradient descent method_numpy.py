# 导入numpy库，用于数值计算
import numpy as np
# 导入matplotlib.pyplot模块，用于绘图
import matplotlib.pyplot as plt

# 定义输入数据x，表示房屋面积
x = np.array([137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00,
              106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21])
# 定义目标数据y，表示房屋价格
y = np.array([145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00,
              62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69, 95.30])

# 设置学习率
learn_rate = 0.00001
# 设置迭代次数
iter = 100
# 设置显示步长，每display_step次迭代输出一次结果
display_step = 10

# 设置随机种子，确保结果可复现
np.random.seed(612)
# 随机初始化权重w
w = np.random.randn()
# 随机初始化偏置b
b = np.random.randn()

# 初始化均方误差列表，用于记录每次迭代的损失
mse = []

# 开始梯度下降迭代
for i in range(0, iter + 1):
    # 计算关于w的梯度：dL_dw = mean(x * (wx + b - y))
    dL_dw = float(np.mean(x * (w * x + b - y)))
    # 计算关于b的梯度：dL_db = mean(wx + b - y)
    dL_db = float(np.mean(w * x + b - y))

    # 更新权重w：w = w - learn_rate * dL_dw
    w = w - learn_rate * dL_dw
    # 更新偏置b：b = b - learn_rate * dL_db
    b = b - learn_rate * dL_db

    # 计算当前预测值p：p = wx + b
    p = w * x + b
    # 计算当前损失：Loss = mean((y - p)^2) / 2
    Loss = np.mean(np.square(y - p)) / 2
    # 将当前损失值添加到mse列表中
    mse.append(Loss)

    # 每display_step次迭代输出一次结果
    if i % display_step == 0:
        print("i: %i, Loss:%f, w: %f, b: %f" % (i, mse[i], w, b))
        print("p:", p)
# 重新计算最终预测值p
p = w * x + b
print("p2:", p)

# 设置matplotlib字体为黑体，用于显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']

# 创建一个新的图形
plt.figure()

# 绘制原始销售记录散点图，颜色为红色
plt.scatter(x, y, color="red", label="销售记录")
# 绘制预测结果散点图，颜色为蓝色
plt.scatter(x, p, color="blue", label="梯度下降法")
# 绘制预测结果的连线，颜色为蓝色
plt.plot(x, p, color="blue")

# 设置x轴标签
plt.xlabel("Area", fontsize=14)
# 设置y轴标签
plt.ylabel("Price", fontsize=14)

# 添加图例
plt.legend(loc="upper left")
# 显示图形
plt.show()

# 创建一个新的图形
plt.figure()
# 绘制损失值随迭代次数的变化曲线
plt.plot(mse)

# 设置x轴标签
plt.xlabel("Iteration", fontsize=14)
# 设置y轴标签
plt.ylabel("Loss", fontsize=14)

# 显示图形
plt.show()

# 创建一个新的图形
plt.figure()
# 绘制从第20次迭代到第100次迭代的损失值变化曲线
plt.plot(range(20, 100), mse[20:100])

# 设置x轴标签
plt.xlabel("Iteration", fontsize=14)
# 设置y轴标签
plt.ylabel("Loss", fontsize=14)

# 显示图形
plt.show()

# 创建一个新的图形
plt.figure()
# 绘制从第40次迭代到第100次迭代的损失值变化曲线
plt.plot(range(40, 100), mse[40:100])

# 设置x轴标签
plt.xlabel("Iteration", fontsize=14)
# 设置y轴标签
plt.ylabel("Loss", fontsize=14)

# 显示图形
plt.show()

# 绘制原始销售记录和预测结果的折线图
plt.plot(y, color="red", marker="o", label="销售记录")
plt.plot(p, color="blue", marker="o", label="梯度下降法")

# 添加图例
plt.legend()
# 设置x轴标签
plt.xlabel("Sample", fontsize=14)
# 设置y轴标签
plt.ylabel("PRICE", fontsize=14)
# 显示图形
plt.show()

# 绘制原始销售记录、预测结果和解析法结果的折线图
plt.plot(y, color="red", marker="o", label="销售记录")
plt.plot(p, color="blue", marker="o", label="梯度下降法")
# 绘制解析法的预测结果，假设解析法的模型为 y = 0.89x + 5.41
plt.plot(0.89 * x + 5.41, color="green", marker="o", label="解析法")

# 添加图例
plt.legend()
# 设置x轴标签
plt.xlabel("Sample", fontsize=14)
# 设置y轴标签
plt.ylabel("PRICE", fontsize=14)
# 显示图形
plt.show()

# 设置matplotlib字体为黑体，用于显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']

# 创建一个新的图形，设置总宽度为20，高度为4
plt.figure(figsize=(20, 4))

# 在1x3的子图布局中绘制第1个子图
plt.subplot(1, 3, 1)
# 绘制原始销售记录散点图和预测结果的连线
plt.scatter(x, y, color="red", label="销售记录")
plt.plot(x, p, color="blue", label="预测模型")
# 设置x轴标签
plt.xlabel("Area", fontsize=14)
# 设置y轴标签
plt.ylabel("PRICE", fontsize=14)
# 添加图例
plt.legend(loc="upper left")

# 在1x3的子图布局中绘制第2个子图
plt.subplot(1, 3, 2)
# 绘制损失值随迭代次数的变化曲线
plt.plot(mse)
# 设置x轴标签
plt.xlabel("Iteration", fontsize=14)
# 设置y轴标签
plt.ylabel("Loss", fontsize=14)

# 在1x3的子图布局中绘制第3个子图
plt.subplot(1, 3, 3)
# 绘制原始销售记录和预测结果的折线图
plt.plot(y, color="red", marker="o", label="销售记录")
plt.plot(p, color="blue", marker="o", label="预测房价")
# 添加图例
plt.legend()
# 设置x轴标签
plt.xlabel("Sample", fontsize=14)
# 设置y轴标签
plt.ylabel("PRICE", fontsize=14)
# 添加图例
plt.legend(loc="upper left")

# 显示图形
plt.show()