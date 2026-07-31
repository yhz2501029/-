import numpy as np
import torch
import matplotlib.pyplot as plt

# prepare dataset
xy = np.loadtxt('糖尿病数据集/diabetes.csv', delimiter=',', dtype=np.float32,skiprows=1)
x_data = torch.from_numpy(xy[:, :-1])  # 第一个‘：’是指读取所有行，第二个‘：’是指从第一列开始，最后一列不要
y_data = torch.from_numpy(xy[:, [-1]])  # [-1] 最后得到的是个矩阵
print(xy.shape)
print(x_data.shape)
print(y_data.shape)

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(
    x_data,
    y_data,
    test_size=0.2,
    random_state=42
)
# design model using class


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)  # 输入数据x的特征是8维，x有8个特征
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()  # 将其看作是网络的一层，而不是简单的函数使用

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))  # y hat
        return x


model = Model()

# construct loss and optimizer
# criterion = torch.nn.BCELoss(size_average = True)
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epoch_list = []
loss_list = []
# training cycle forward, backward, update
for epoch in range(2000):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
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

with torch.no_grad():

    y_pred=model(x_val)

    pred=(y_pred>0.5).float()

    acc=(pred==y_val).float().mean()

    print("验证准确率:",acc.item())
x_test = torch.Tensor([[4.0, 85.0, 66.0, 29.0, 0.0, 26.6, 0.351, 31.0]])
y_test=model(x_test)
print("y_pred=",y_test.item())