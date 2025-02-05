# 导入matplotlib.pyplot模块，用于绘图
import matplotlib.pyplot as plt
# 导入numpy库，用于数值计算
import numpy as np

# 定义输入数据x1，表示房屋面积
x1 = np.array([137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00,
               106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21])
# 定义输入数据x2，表示房间数
x2 = np.array([3, 2, 2, 3, 1, 2, 3, 2, 2, 3, 1, 1, 1, 1, 2, 2])
# 定义目标数据y，表示房屋价格
y = np.array([145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00,
              62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69, 95.30])

# 定义权重数组W，包含偏置项、x1的权重和x2的权重
W = np.array([11.96729093, 0.53488599, 14.33150378])
# 使用权重数组W计算预测值y_pred
y_pred = W[1] * x1 + W[2] * x2 + W[0]

# 创建一个图形对象，设置图形大小为8x6英寸
fig = plt.figure(figsize=(8, 6))
# 添加一个3D子图
ax3d = fig.add_subplot(111, projection='3d')

# 在3D图中绘制散点图，表示原始数据点
ax3d.scatter(x1, x2, y, color="b", marker="*")

# 设置x轴标签
ax3d.set_xlabel('Area', color='r', fontsize=16)
# 设置y轴标签
ax3d.set_ylabel('Room', color='r', fontsize=16)
# 设置z轴标签
ax3d.set_zlabel('Price', color='r', fontsize=16)
# 设置y轴的刻度
ax3d.set_yticks([1, 2, 3])
# 设置z轴的范围
ax3d.set_zlim3d(30, 160)

# 显示图形
plt.show()

# 创建另一个图形对象，设置图形大小为8x6英寸
fig = plt.figure(figsize=(8, 6))
# 添加一个3D子图
ax3d = fig.add_subplot(111, projection='3d')
# 设置3D图的视角，俯仰角为0度，方位角为-90度
ax3d.view_init(elev=0, azim=-90)

# 在3D图中绘制散点图
ax3d.scatter(x1, x2, y, color='b')
# 设置轴标签
ax3d.set_xlabel('Area', color='r', fontsize=16)
ax3d.set_ylabel('Room', color='r', fontsize=16)
ax3d.set_zlabel('Price', color='r', fontsize=16)
# 设置y轴的刻度
ax3d.set_yticks([1, 2, 3])
# 设置z轴的范围
ax3d.set_zlim3d(30, 160)

# 显示图形
plt.show()

# 创建第三个图形对象，设置图形大小为8x6英寸
fig = plt.figure(figsize=(8, 6))
# 添加一个3D子图
ax3d = fig.add_subplot(111, projection='3d')
# 设置3D图的视角，俯仰角为0度，方位角为0度
ax3d.view_init(elev=0, azim=0)

# 在3D图中绘制散点图
ax3d.scatter(x1, x2, y, color='b')

# 设置轴标签
ax3d.set_xlabel('Area', color='r', fontsize=16)
ax3d.set_ylabel('Room', color='r', fontsize=16)
ax3d.set_zlabel('Price', color='r', fontsize=16)
# 设置y轴的刻度
ax3d.set_yticks([1, 2, 3])
# 设置z轴的范围
ax3d.set_zlim3d(30, 160)

# 显示图形
plt.show()

# 使用numpy的meshgrid函数生成x1和x2的网格数据
X1, X2 = np.meshgrid(x1, x2)
# 计算预测值的网格数据
Y_PRED = W[0] + W[1] * X1 + W[2] * X2

# 创建一个新的图形对象
fig = plt.figure()
# 添加一个3D子图
ax3d = fig.add_subplot(111, projection='3d')

# 在3D图中绘制预测值的曲面图，使用coolwarm颜色映射
ax3d.plot_surface(X1, X2, Y_PRED, cmap="coolwarm")

# 设置轴标签
ax3d.set_xlabel('Area', color='r', fontsize=14)
ax3d.set_ylabel('Room', color='r', fontsize=14)
ax3d.set_zlabel('Price', color='r', fontsize=14)
# 设置y轴的刻度
ax3d.set_yticks([1, 2, 3])

# 显示图形
plt.show()

# 设置matplotlib的字体为黑体，用于显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']

# 创建一个新的图形对象
fig = plt.figure()
# 添加一个3D子图
ax3d = fig.add_subplot(111, projection='3d')
# 绘制原始数据点的散点图，颜色为蓝色，标记为星号
ax3d.scatter(x1, x2, y, color='b', marker="*", label="销售记录")
# 绘制预测值的散点图，颜色为红色
ax3d.scatter(x1, x2, y_pred, color='r', label="预测房价")
# 绘制预测值的网格线图，颜色为青色，线宽为0.5
ax3d.plot_wireframe(X1, X2, Y_PRED, color='c', linewidth=0.5, label="拟合平面")

# 设置轴标签
ax3d.set_xlabel('Area', color='r', fontsize=14)
ax3d.set_ylabel('Room', color='r', fontsize=14)
ax3d.set_zlabel('Price', color='r', fontsize=14)
# 设置y轴的刻度
ax3d.set_yticks([1, 2, 3])

# 设置图形的总标题
plt.suptitle("商品房销售回归模型", fontsize=20)
# 添加图例
plt.legend(loc="upper left")
# 显示图形
plt.show()