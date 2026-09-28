import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import matplotlib.pyplot as plt

import pandas as pd
import torch

# 读取数据
df = pd.read_csv('泰坦尼克号数据集/titanic/train.csv')
tf=pd.read_csv('泰坦尼克号数据集/titanic/test.csv')
# ====================
# 取标签
# ====================

y_data = torch.tensor(
    df['Survived'].values,
    dtype=torch.float32
).reshape(-1,1)


# ====================
# 取特征
# ====================

x = df[
    [
        'Pclass',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Fare',
        'Embarked'
    ]
].copy()


# ====================
# 缺失值处理
# ====================

# Age缺失
x['Age'] = x['Age'].fillna(
    x['Age'].mean()
)


# Embarked缺失
x['Embarked'] = x['Embarked'].fillna(
    x['Embarked'].mode()[0]
)


# ====================
# 字符串转数字
# ====================

# Sex
x['Sex'] = x['Sex'].map(
    {
        'male':0,
        'female':1
    }
)


# Embarked
x['Embarked'] = x['Embarked'].map(
    {
        'S':0,
        'C':1,
        'Q':2
    }
)


# ====================
# 转Tensor
# ====================

x_data = torch.tensor(
    x.values,
    dtype=torch.float32
)
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
        self.linear1 = torch.nn.Linear(7, 6)  # 输入数据x的特征是8维，x有8个特征
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
# ==========================
# 处理test数据
# ==========================

test_x = tf[
    [
        'Pclass',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Fare',
        'Embarked'
    ]
].copy()


# Age缺失
test_x['Age'] = test_x['Age'].fillna(
    x['Age'].mean()
)


# Fare缺失
test_x['Fare'] = test_x['Fare'].fillna(
    x['Fare'].mean()
)


# Embarked缺失
test_x['Embarked'] = test_x['Embarked'].fillna(
    x['Embarked'].mode()[0]
)


# ==========================
# 字符串编码
# ==========================

test_x['Sex'] = test_x['Sex'].map(
    {
        'male':0,
        'female':1
    }
)


test_x['Embarked'] = test_x['Embarked'].map(
    {
        'S':0,
        'C':1,
        'Q':2
    }
)


# ==========================
# 标准化
# 注意：
# 这里不能重新fit
# 必须使用训练集的scaler
# ==========================

test_x = torch.tensor(
    scaler.transform(test_x),
    dtype=torch.float32
)


print(test_x.shape)
# ==========================
# 模型预测
# ==========================

model.eval()

with torch.no_grad():

    pred = model(test_x)


# 概率转0/1
pred = (pred >= 0.5).int()


print(pred[:10])
# ==========================
# 保存Kaggle格式
# ==========================


submission = pd.DataFrame(
    {
        "PassengerId": tf["PassengerId"],
        "Survived": pred.numpy().reshape(-1)
    }
)


submission.to_csv(
    r"D:\Pycharm\深度学习\outputs\泰坦尼克\submission.csv",
    index=False
)


print(submission.head())