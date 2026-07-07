
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ================= 1. 设定超参数与生成人工数据集 =================
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
# 特征维度(200)远大于训练样本数(20)，这是典型的“高维小样本”过拟合场景

true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05#torch.one用于生成每个元素都为1的向量

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声 (替代 d2l.synthetic_data)"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # 添加均值为0、标准差为0.01的高斯噪声
    return X, y.reshape((-1, 1))

train_features, train_labels = synthetic_data(true_w, true_b, n_train)
train_dataset = TensorDataset(train_features, train_labels)
train_iter = DataLoader(train_dataset, batch_size, shuffle=True)

test_features, test_labels = synthetic_data(true_w, true_b, n_test)
test_dataset = TensorDataset(test_features, test_labels)
test_iter = DataLoader(test_dataset, batch_size, shuffle=False)

def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)#创建初始为0的向量，
    return [w, b]
def linreg(X, w, b):
    """线性回归模型 (替代 d2l.linreg)"""
    return torch.matmul(X, w) + b#用于生成y_hat

def squared_loss(y_hat, y):
    """平方损失函数 (替代 d2l.squared_loss)"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def l2_penalty(w):
    """定义 L2 范数惩罚项（权重衰减）"""
    return torch.sum(w.pow(2)) / 2

# ================= 4. 优化器与评估函数 =================
def sgd(params, lr, batch_size):
    """小批量随机梯度下降 (替代 d2l.sgd)"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def evaluate_loss(net, data_iter, loss, w, b):
    """评估模型在指定数据集上的平均损失 (替代 d2l.evaluate_loss)"""
    metric_loss = 0.0
    metric_count = 0
    with torch.no_grad():
        for X, y in data_iter:
            out = net(X, w, b)
            l = loss(out, y)
            metric_loss += l.sum().item()#item表示把张量的括号去掉，变成一个数字
            metric_count += l.numel() #numel会返回张量中的元素个数，如果是二维会返回形状
    return metric_loss / metric_count


def train(lambd):
    w, b = init_params()
    net = linreg
    loss = squared_loss
    num_epochs, lr = 100, 0.003

    # 用于记录绘图数据（替代 d2l.Animator）
    epochs_list = []
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 核心：在传统损失中加入 L2 惩罚项。lambd 控制惩罚力度
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.sum().backward()
            sgd([w, b], lr, batch_size)

        # 每 5 个 epoch 记录一次损失
        if (epoch + 1) % 5 == 0:
            epochs_list.append(epoch + 1)
            train_loss = evaluate_loss(net, train_iter, loss, w, b)
            test_loss = evaluate_loss(net, test_iter, loss, w, b)
            train_losses.append(train_loss)#把元素添加到列表的末尾
            test_losses.append(test_loss)

    # 使用 Matplotlib 统一绘制训练与测试的损失曲线
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_list, train_losses, label='train loss')
    plt.plot(epochs_list, test_losses, linestyle='--', label='test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')  # 使用对数纵坐标，便于看清低值时的差距
    plt.xlim([5, num_epochs])
    plt.legend()
    plt.grid(True)
    plt.show()

    print('w的L2范数是：', torch.norm(w).item())




train(lambd=6)