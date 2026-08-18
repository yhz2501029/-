import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import matplotlib.pyplot as plt

import pandas as pd
import torch

# 读取数据
df = pd.read_csv('鸢尾花数据集/iris/iris.data')

# ====================
# 取标签
# ====================

y_data = df.iloc[:, -1].map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
})
# ====================
# 取特征
# ====================

x = df.iloc[:, :4].copy()

# ====================
# 字符串转数字
# ====================
y_data = torch.tensor(y_data.values, dtype=torch.long)
# ====================
# 转Tensor
# ====================

x_data = torch.tensor(x.values,dtype=torch.float32)
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()

x_data = torch.tensor(
    scaler.fit_transform(x_data),
    dtype=torch.float32
)

print(x_data.shape)
print(y_data.shape)
print(x_data[:5])
print(y_data[:5])
# design model using class


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(4, 3)  # 输入数据x的特征是8维，x有8个特征
        self.sigmoid = torch.nn.Sigmoid()  # 将其看作是网络的一层，而不是简单的函数使用

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))  # y hat
        return x


model = Model()

# construct loss and optimizer
# criterion = torch.nn.BCELoss(size_average = True)
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epoch_list = []
loss_list = []
# training cycle forward, backward, update
for epoch in range(2000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
