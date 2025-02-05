# 导入TensorFlow库，用于构建和训练模型
import tensorflow as tf
# 打印TensorFlow的版本信息，确保安装的版本正确
print("TensorFlow version: ", tf.__version__)
# 导入NumPy库，用于处理数组和数学运算
import numpy as np

# 定义输入数据x，表示房屋面积（单位：平方米）
x = np.array([137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00,
              106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21])
# 定义目标数据y，表示房屋价格（单位：万元）
y = np.array([145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00,
              62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69, 95.30])

# 定义学习率，控制模型参数更新的步长
learn_rate = 0.0001
# 定义迭代次数，即训练模型的轮数
iter = 10
# 定义显示步长，用于在训练过程中打印中间结果
display_step = 1

# 设置随机种子，确保每次运行代码时生成的随机数相同
np.random.seed(612)
# 初始化模型参数w（权重）和b（偏置），使用随机值
w = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

# 用于存储每轮迭代的均方误差（MSE）
mse = []

# 开始训练模型，循环迭代指定次数
for i in range(0, iter + 1):
    # 使用GradientTape记录计算过程，以便后续计算梯度
    with tf.GradientTape() as tape:
        # 计算模型的预测值，线性回归公式：y = w * x + b
        pred = w * x + b
        # 计算损失函数，使用均方误差（MSE）作为损失函数
        Loss = tf.reduce_mean(tf.square(y - pred))
    # 将每轮的损失值存储到mse列表中
    mse.append(Loss)

    # 计算损失函数关于w和b的梯度
    dL_dw, dL_db = tape.gradient(Loss, [w, b])

    # 使用梯度下降法更新模型参数w和b
    w.assign_sub(learn_rate * dL_dw)
    b.assign_sub(learn_rate * dL_db)

    # 每隔display_step步长打印一次训练结果
    if i % display_step == 0:
        print("i: %i，Loss: %f, w: %f b: %f" % (i, Loss.numpy(), w.numpy(), b.numpy()))