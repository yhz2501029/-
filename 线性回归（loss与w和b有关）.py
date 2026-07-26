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

# 3. 定义网格范围 (用于绘图)
# w 的范围设为 0.0 ~ 4.1，b 的范围设为 -2.0 ~ 2.0
w_range = np.arange(0.0, 4.1, 0.1)
b_range = np.arange(-2.0, 2.1, 0.1)

# 使用 meshgrid 生成网格矩阵
W, B = np.meshgrid(w_range, b_range)

# 4. 计算每个 (w, b) 组合对应的总 Loss
# 初始化一个与 W 形状相同的零矩阵来存放 Loss
Loss_Matrix = np.zeros_like(W)

# 遍历所有数据点计算 Loss
# 注意：这里为了计算整个曲面的高度，需要对每一个 w 和 b 的组合都算一遍
for i in range(len(x_data)):
    x_val = x_data[i]
    y_val = y_data[i]
    # 向量化计算：直接对整个矩阵进行运算，比双重 for 循环快得多
    y_pred = x_val * W + B
    Loss_Matrix += (y_pred - y_val) ** 2

# 求平均 MSE (除以数据点个数 3)
MSE_Matrix = Loss_Matrix / len(x_data)

# 5. 绘制三维图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
# cmap='rainbow' 设置颜色映射，edgecolor='none' 让曲面更平滑
surf = ax.plot_surface(W, B, MSE_Matrix, cmap='rainbow', edgecolor='none', alpha=0.9)

# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=5)

# 设置标签
ax.set_xlabel('Weight (w)')
ax.set_ylabel('Bias (b)')
ax.set_zlabel('Loss (MSE)')
ax.set_title('Loss Surface for y = wx + b')

plt.show()