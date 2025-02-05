import numpy as np
import matplotlib.pyplot as plt

x = np.array([137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00,
              106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21])
y = np.array([145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00,
              62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69, 95.30])

learn_rate = 0.00001
iter = 100
display_step = 10

np.random.seed(612)
w = np.random.randn()
b = np.random.randn()

mse = []

for i in range(0, iter + 1):
    dL_dw = float(np.mean(x * (w * x + b - y)))
    dL_db = float(np.mean(w * x + b - y))

    w = w - learn_rate * dL_dw
    b = b - learn_rate * dL_db

    p = w * x + b
    Loss = np.mean(np.square(y - p)) / 2
    mse.append(Loss)

    if i % display_step == 0:
        print("i: %i, Loss:%f, w: %f, b: %f" % (i, mse[i], w, b))
        print("p:",p)
# 重新计算 p 的值
p = w * x + b
print("p2:",p)
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure()

plt.scatter(x, y, color="red", label="销售记录")
plt.scatter(x, p, color="blue", label="梯度下降法")
plt.plot(x, p, color="blue")

plt.xlabel("Area", fontsize=14)
plt.ylabel("Price", fontsize=14)

plt.legend(loc="upper left")
plt.show()

plt.figure()
plt.plot(mse)

plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Loss", fontsize=14)

plt.show()

plt.figure()

plt.plot(range(20, 100), mse[20:100])

plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Loss", fontsize=14)

plt.show()

plt.figure()

plt.plot(range(40, 100), mse[40:100])

plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Loss", fontsize=14)

plt.show()

plt.plot(y, color="red", marker="o", label="销售记录")
plt.plot(p, color="blue", marker="o", label="梯度下降法")

plt.legend()
plt.xlabel("Sample", fontsize=14)
plt.ylabel("PRICE", fontsize=14)
plt.show()

plt.plot(y, color="red", marker="o", label="销售记录")
plt.plot(p, color="blue", marker="o", label="梯度下降法")
plt.plot(0.89*x+5.41, color="green", marker="o", label="解析法")

plt.legend()
plt.xlabel("Sample", fontsize=14)
plt.ylabel("PRICE", fontsize=14)
plt.show()

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure(figsize=(20, 4))

plt.subplot(1, 3, 1)
plt.scatter(x, y, color="red", label="销售记录")
plt.plot(x, p, color="blue", label="预测模型")
plt.xlabel("Area", fontsize=14)
plt.ylabel("PRICE", fontsize=14)
plt.legend(loc="upper left")

plt.subplot(1, 3, 2)
plt.plot(mse)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Loss", fontsize=14)

plt.subplot(1, 3, 3)
plt.plot(y, color="red", marker="o", label="销售记录")
plt.plot(p, color="blue", marker="o", label="预测房价")
plt.legend()
plt.xlabel("Sample", fontsize=14)
plt.ylabel("PRICE", fontsize=14)
plt.legend(loc="upper left")

plt.show()

