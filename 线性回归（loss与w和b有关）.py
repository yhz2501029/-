import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 准备数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# 2. 定义模型和损失函数
def forward(x, w, b):
    return x * w + b


def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2


# 3. 定义参数范围
w_range = np.arange(0.0, 4.1, 0.1)  # w 从 0 到 4
b_range = np.arange(-2.0, 2.1, 0.1)  # b 从 -2 到 2

# 4. 生成网格矩阵（仅用于最后绘图，不参与 Loss 计算）
W, B = np.meshgrid(w_range, b_range)
Loss_Matrix = np.zeros_like(W, dtype=float)

# 5. 核心：三重循环计算 Loss
print("开始暴力计算...")
for i, w in enumerate(w_range):  # 第一重：遍历 w
    for j, b in enumerate(b_range):  # 第二重：遍历 b
        total_loss = 0
        for x_val, y_val in zip(x_data, y_data):  # 第三重：遍历数据点
            total_loss += loss(x_val, y_val, w, b)

        # 计算 MSE 并存入矩阵
        # 注意：W 和 B 的形状是 (len(b), len(w))，所以索引是 [j, i]
        Loss_Matrix[j, i] = total_loss / len(x_data)

print("计算完成，正在绘图...")

# 6. 绘制三维图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(W, B, Loss_Matrix, cmap='rainbow', edgecolor='none', alpha=0.9)
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel('Weight (w)')
ax.set_ylabel('Bias (b)')
ax.set_zlabel('Loss (MSE)')
ax.set_title('Loss Surface (Triple Loop Version)')

plt.show()