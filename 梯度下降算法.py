import matplotlib.pyplot as plt  # 1. 导入绘图库

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


print('Predict (before training)', 4, forward(4))

# 2. 创建列表用于存储每一轮的数据
epoch_list = []
cost_list = []

for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val

    # 3. 记录当前轮次和损失值
    epoch_list.append(epoch)
    cost_list.append(cost_val)

    print('Epoch:', epoch, 'w=', w, 'loss=', cost_val)

print('Predict (after training)', 4, forward(4))

# 4. 训练结束后绘图
plt.plot(epoch_list, cost_list)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss Curve')
plt.grid(True)  # 添加网格线方便观察
plt.show()