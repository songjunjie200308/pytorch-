import torch
import torchvision
from torch.utils import data
from torchvision import transforms
# ==================== 1. 数据加载模块 ====================
def get_dataloader_workers():
    # 为了绝对安全（防止Windows系统多进程卡死），先设为0（单进程读取）
    return 0
def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="01_data/01_DataSet_FashionMNIST", train=True, transform=trans,
                                                    download=True)
    # 🌟 修复：这里改为了 mnist_test，加载真实的测试集
    mnist_test = torchvision.datasets.FashionMNIST(root="01_data/01_DataSet_FashionMNIST", train=False, transform=trans,
                                                   download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))
# ==================== 2. 模型与核心算法定义 ====================
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition
def net(X):
    X_flat = X.reshape((-1, w.shape[0]))#-1表示自动计算一个批的数量是多少，w是一个形状为728x10的矩阵，
    linear_output = torch.matmul(X_flat, w)#xflat是一个batchsize*728的矩阵
    logits = linear_output + b#这里有自动补齐机制，会把b从1*10的的变成256*10的
    return softmax(logits)
def cross_entropy(y_hat, y ):
    batch_size = y_hat.shape[0]
    row_indices = range(batch_size)
    col_indices = y
    correct_class_probs = y_hat[row_indices, col_indices]#运行的结果是一个张量，把每个正确值对应的概率拿出来，排成一行
    return -torch.log(correct_class_probs)#记录每个batch中的每个样本的损失
# ==================== 3. 训练与评估内核 ====================
def evaluate_accuracy_simple(net, data_iter):
    """极其通俗易懂的准确率计算函数"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    total_correct = 0.0
    total_samples = 0
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            batch_correct = (y_hat.argmax(dim=1) == y).type(torch.float32).sum().item()
            batch_total = y.numel()
            total_correct += batch_correct
            total_samples += batch_total
    return total_correct / total_samples
def train_epoch_ch3_simple(net, train_iter, loss, updater):
    """单轮训练逻辑"""
    if isinstance(net, torch.nn.Module):
        net.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        preds = y_hat.argmax(dim=1)#生成一行，长度为一个batch_size
        batch_correct = (preds == y).type(torch.float32).sum().item()
        batch_total = y.numel()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
            batch_loss = float(l) * batch_total
        else:
            l.sum().backward()
            updater(X.shape[0])
            batch_loss = float(l.sum())
        total_loss += batch_loss
        total_correct += batch_correct
        total_samples += batch_total
    return total_loss / total_samples, total_correct / total_samples


def train_ch3_simple(net, train_iter, test_iter, loss, num_epochs, updater):
    """总训练函数：掌控多轮训练与评估的全局流程"""
    print(f"🎬 开始训练流程，总计 {num_epochs} 轮次...")
    print("=" * 60)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_ch3_simple(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy_simple(net, test_iter)

        print(f" Epoch [{epoch + 1:02d}/{num_epochs:02d}] 战报:")
        print(f"   - 📉 训练集平均损失 (Loss): {train_loss:.4f}")
        print(f"   - 🎯 训练集准确率 (Train Acc): {train_acc * 100:.2f}%")
        print(f"   - 🧪 测试集准确率 (Test Acc):  {test_acc * 100:.2f}%")
        print("-" * 60)

    print("🏆 训练全部结束！模型已整装待发。")


# ==================== 4. 真实全局执行空间 ====================
# 🌟 注意看：这里的代码完全顶格写，不再缩进在任何函数里面了！

if __name__ == '__main__':
    # 初始化数据迭代器
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    # 初始化模型权重和偏置
    num_inputs = 784
    num_outputs = 10
    w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    # 定义超参数
    lr = 0.1
    num_epochs = 5


    # 定义手写优化器
    def updater(batch_size):
        global w, b  # 🌟 修复：声明全局变量，否则无法直接修改外面的 w 和 b
        with torch.no_grad():
            w -= lr * w.grad / batch_size
            b -= lr * b.grad / batch_size
            w.grad.zero_()
            b.grad.zero_()


    # 🚀 点火，起航！
    train_ch3_simple(net, train_iter, test_iter, cross_entropy, num_epochs, updater)