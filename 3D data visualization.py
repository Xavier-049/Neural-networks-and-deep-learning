# 导入matplotlib.pyplot模块，用于绘图
import matplotlib.pyplot as plt

# 创建一个图形对象
fig = plt.figure()
# 添加一个3D子图
ax3d = fig.add_subplot(111, projection='3d')
# 显示图形
plt.show()

# 导入numpy库，用于数值计算
import numpy as np

# 使用numpy生成30个在[10, 40)区间内的随机数作为x坐标
x = np.random.uniform(10, 40, 30)
# 使用numpy生成30个在[100, 200)区间内的随机数作为y坐标
y = np.random.uniform(100, 200, 30)
# 使用numpy生成30个在[10, 20)区间内的随机数作为z坐标
z = np.random.uniform(10, 20, 30)

# 创建一个图形对象
fig = plt.figure()
# 添加一个3D子图
ax3d = fig.add_subplot(111, projection='3d')
# 在3D图中绘制散点图，颜色为蓝色，形状为星号
ax3d.scatter(x, y, z, c='b', marker="*")
# 显示图形
plt.show()

# 使用numpy生成30个在[10, 40)区间内的随机数作为x坐标
x = np.random.uniform(10, 40, 30)
# 使用numpy生成30个在[100, 200)区间内的随机数作为y坐标
y = np.random.uniform(100, 200, 30)
# 计算z坐标，z = 2x + y
z = 2 * x + y

# 创建一个图形对象
fig = plt.figure()
# 添加一个3D子图
ax3d = fig.add_subplot(111, projection='3d')

# 在3D图中绘制散点图，颜色为蓝色，形状为星号
ax3d.scatter(x, y, z, c='b', marker="*")

# 设置x轴标签
ax3d.set_xlabel('X')
# 设置y轴标签
ax3d.set_ylabel('Y')
# 设置z轴标签
ax3d.set_zlabel('Z=2X+Y')

# 显示图形
plt.show()

# 使用numpy生成300个在[10, 40)区间内的随机数作为x坐标
x = np.random.uniform(10, 40, 300)
# 使用numpy生成300个在[100, 200)区间内的随机数作为y坐标
y = np.random.uniform(100, 200, 300)
# 计算z坐标，z = 2x + y
z = 2 * x + y

# 创建一个图形对象
fig = plt.figure()
# 添加一个3D子图
ax3d = fig.add_subplot(111, projection='3d')

# 在3D图中绘制散点图，颜色为蓝色，形状为星号
ax3d.scatter(x, y, z, c='b', marker="*")

# 设置x轴标签
ax3d.set_xlabel('X')
# 设置y轴标签
ax3d.set_ylabel('Y')
# 设置z轴标签
ax3d.set_zlabel('Z=2X+Y')

# 显示图形
plt.show()
