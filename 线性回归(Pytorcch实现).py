import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)#self.linear是一个对象，torch.nn.Linear是一个类

    def forward(self, x):
        y_pred = self.linear(x)#对象(x)会自动调用__call__函数，nn.Module类里面的__call__函数里面有个叫forward的函数，linear类也一样，linear类的对象(x)也会调用call函数
        return y_pred


model = LinearModel()

criterion = torch.nn.MSELoss(size_average=False)#同上，criterion也是一个对象
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)#同上

for epoch in range(1000):
    y_pred = model(x_data)#自动调用__call__函数
    loss = criterion(y_pred, y_data)#自动调用__call__函数
    print(epoch, loss.item())

    optimizer.zero_grad()#未调用__call__函数，就是调用的.zero_grad()函数
    loss.backward()
    optimizer.step()#未调用__call__函数，调用.step函数，去更新可更新的权重，所利用的梯度是loss.backward()求出来的

print('w = ', model.linear.weight.item())#model对象下有linear对象，linear对象有w这个张量
print('b = ', model.linear.bias.item())#model对象下有linear对象，linear对象有b这个张量

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)