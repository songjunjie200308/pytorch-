import numpy as np
import torch
from torch.utils import data
# from d2l import torch as d2l
from torch import nn
def synthetic_data(w, b, num_examples): # 顺便帮你修正了原代码的拼写错误 exaples -> examples
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))#norm函数用于生成符合正态分布的随机数，三个参数分别为均值标准差形状，num_examples规定生成的行,len代表列
    y = torch.matmul(X, w) + b#matmual是矩阵乘法的缩写，意思是矩阵乘法
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))#-1表示自动计算，在只有一列的情况下，一共可以有多少行来组成一列，在这里相当于把所有向量排成一列

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)  # 库函数生成人工数据集


# 调用框架现有的API来读取数据
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个Pytorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)  # dataset相当于Pytorch的Dataset。一个星号*，表示对list解开入参。
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # 返回的是从dataset中随机挑选出batch_size个样本出来


batch_size = 10
data_iter = load_array((features, labels), batch_size)  # 返回的数据的迭代器
print(next(iter(data_iter)))  # iter(data_iter) 是一个迭代器对象，next是取迭代器里面的元素

# 使用框架的预定义好的层
# nn是神经网络的缩写
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)  # net[i]表示sequential的第i层，weight表示该层的权重矩阵，data表示修改其中的参数，nomal表示修改方式
net[0].bias.data.fill_(0)  # 同理，此处的bias表示对应层的偏置向量，fill表示将偏置值改为对应的数
print(net[0])

# 计算均方误差使用的是MSELoss类，也称为平方L2范数
loss = nn.MSELoss()  # L1是算术差，L2是平方差

# 实例化SGD实例
trainer = torch.optim.SGD(net.parameters(), lr=0.03)#parameters会将该线性层的函数全部收集在一起，形成一个参数集合

# 训练过程代码与从零开始时所做的非常相似
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:  # 从DataLoader里面一次一次把所有数据拿出来
#         print("X:",X)
#         print("y:",y)
        l = loss(net(X),y) # net(X) 为计算出来的线性回归的预测值
        trainer.zero_grad() # 梯度清零
        l.backward()
        trainer.step()  # SGD优化器优化模型
# 手动更新版本如下
# with torch.no_grad():
#     for param in params:
#         param -= lr * param.grad / batch_size
    l = loss(net(features),labels)
    print(f'epoch{epoch+1},loss{l:f}')