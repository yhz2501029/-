import torch

batch_size = 1
input_size = 4
hidden_size = 4
num_layers = 1
seq_len = 5
idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]  # hello
y_data = [3, 1, 2, 3, 2]  # ohlol
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data)  # (seqlen×batchsize,1)


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(self.input_size, self.hidden_size, self.num_layers)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)  # rashpe out to (seq_len×batch_size, hiddensize)


net = Model(input_size, hidden_size, batch_size, num_layers)
critirion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = critirion(outputs, labels)
    loss.backward()
    optimizer.step()

    print('outputs:', outputs)
    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()  # reshape to numpy
    print('idx', idx)
    print('Pridected:', ''.join([idx2char[x] for x in idx]), end='')  # end是不自动换行，''.join是连接字符串数组
    print(',Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))