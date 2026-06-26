import random
import torch
import matplotlib.pyplot as plt  # 1. 导入标准的 matplotlib 绘图库

def synthetic_data(w, b, num_examples): # 顺便帮你修正了原代码的拼写错误 exaples -> examples
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))#norm函数用于生成符合正态分布的随机数，三个参数分别为均值标准差形状，num_examples规定生成的个数
    y = torch.matmul(X, w) + b#matmual是矩阵乘法的缩写，意思是矩阵乘法
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))#-1表示自动计算，在只有一列的情况下，一共可以有多少行来组成一列，在这里相当于把所有向量排成一列

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0], '\nlabel:', labels[0])

# 2. 替换掉 d2l 的画图部分
plt.figure(figsize=(5, 3.5))  # 替代 d2l.set_figsize()，设置图片宽 5 吋，高 3.5 吋
plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1) # 替代 d2l.plt.scatter
plt.xlabel('X')
plt.ylabel('Y')
plt.show()                    # 核心：在 .py 文件里，必须加这一行图片才会弹出来！


# data_iter 函数原本就是纯 PyTorch 写的，完全不需要变动
def data_iter(batch_size, features, labels):
    num_examples = len(features)  # 样本个数
    indices = list(range(num_examples))  # 样本索引
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)  # 把索引随机打乱
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)])  # 当i+batch_size超出时，取num_examples
        yield features[batch_indices], labels[batch_indices]  # 获得随机顺序的特征，及对应的标签


batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print("\n--- 读取到的第一个小批量数据 ---")
    print("X:\n", X)
    print("y:\n", y)
    break