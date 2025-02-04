import tensorflow as tf  # 导入TensorFlow库，并将其命名为tf
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，并将其命名为plt

# 加载波士顿房价数据集
boston_housing = tf.keras.datasets.boston_housing
# 将数据集分为训练集和测试集，这里只取训练集
(train_x, train_y), (_, _) = boston_housing.load_data(test_split=0)

# 设置字体为SimHei，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置坐标轴中的负号正常显示
plt.rcParams['axes.unicode_minus'] = False

# 定义数据集中的各个特征名称
titles = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
          "DIS", "RAD", "TAX", "PTRATIO", "B-1000", "LSTAT", "MEDV"]

# 创建一个新的图形，大小为12x12英寸
plt.figure(figsize=(12, 12))

# 循环遍历13个特征
for i in range(13):
    # 创建一个4x4的子图布局中的第i+1个子图
    plt.subplot(4, 4, (i + 1))
    # 绘制第i个特征与房价的关系散点图
    plt.scatter(train_x[:, i], train_y)
    # 设置x轴标签为当前特征名称
    plt.xlabel(titles[i])
    # 设置y轴标签为“价格（$1000）”
    plt.ylabel("Price($1000's)")
    # 设置子图标题为“第i+1个特征 - 价格”
    plt.title(str(i + 1) + "." + titles[i] + " - Price")

# 自动调整子图参数，使之填充整个图像区域
plt.tight_layout()
# 设置图形的总标题为“各个属性与房价的关系”，位置在图形中央偏上
plt.suptitle("各个属性与房价的关系", x=0.5, y=1.02, fontsize=20)
# 显示图形
plt.show()











