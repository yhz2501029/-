import torch
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
w1=torch.Tensor([1.0])#tensor也叫张量，是由datas和grad（梯度）组成的
w2=torch.Tensor([2.0])
b=torch.Tensor([3.0])
w1.requires_grad=True#需要计算梯度，默认tensor不计算梯度
w2.requires_grad=True
b.requires_grad=True
def forward(x):
    return x*x*w1+x*w2+b
def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)**2
for epoch in range(2000):
    for x,y in zip(x_data,y_data):
        #构建计算图
        l=loss(x,y)#l是张量
        l.backward()#backward:是tensor自带的函数，可以自动计算计算图上所有梯度，存到w中并释放计算图
        print('\tgrad:',x,y,w1.grad.item(),w2.grad.item(),b.grad.item())#用.item是因为grad也是张量，我们想输出标量，所以要这样
        w1.data=w1.data-0.01*w1.grad.data#取data，因为grad（梯度）也是张量，我们想做的是纯数值的计算。
        w2.data=w2.data-0.01*w2.grad.data
        b.data=b.data-0.01*b.grad.data
        w1.grad.data.zero_()#w更新之后给他清零，不清零就会和上次梯度一起相加
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print('progress:',epoch,l.item())
print("predict(after training)",4,forward(4).item())#输入4的结果