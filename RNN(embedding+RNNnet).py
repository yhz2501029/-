import torch

num_class = 4  # 4个类别，
input_size = 4  # 输入维度
hidden_size = 8  # 隐层输出维度，有8个隐层
embedding_size = 10  # 嵌入到10维空间
num_layers = 2  # 2层的RNN
batch_size = 1
seq_len = 5  # 序列长度5

idx2char = ['e', 'h', 'l', 'o']
x_data = [[1, 0, 2, 2, 3]]  # (batch, seq_len) list
y_data = [3, 1, 2, 3, 2]  # (batch * seq_len)
inputs = torch.LongTensor(x_data)  # tensor
labels = torch.LongTensor(y_data)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x)  # (batch,seqlen,embeddingSize)
        x, _ = self.rnn(x, hidden)
        x = self.fc(x)
        return x.view(-1, num_class)  # 矩阵（batchsize×seqlen, numclass）


net = Model()
critirion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = critirion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('idx', idx)
    print('Pridected:', ''.join([idx2char[x] for x in idx]), end='')  # end是不自动换行，''.join是连接字符串数组
    print(',Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))