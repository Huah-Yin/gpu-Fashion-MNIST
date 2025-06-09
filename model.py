import torch
from torch import nn
from d2l import torch as d2l

# ===================== 设备选择 =====================
def get_device():
    """获取可用设备
    
    Returns:
        torch.device: 可用的设备(GPU或CPU)
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== 模型定义 =====================
class FashionMNISTNet(nn.Module):
    """Fashion-MNIST分类网络
    
    改进的多层感知机，包含：
    1. 输入层：784个神经元（28*28像素）
    2. 两个隐藏层：512和256个神经元
    3. 输出层：10个神经元（对应10个类别）
    """
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                           # 将输入展平为向量
            nn.Linear(num_inputs, 512),            # 784 -> 512 第一个全连接层
            nn.ReLU(),                             # ReLU激活函数
            nn.Linear(512, num_hiddens),           # 512 -> 256     第二个全连接层
            nn.ReLU(),                             # ReLU激活函数
            nn.Linear(num_hiddens, num_outputs)    # 256 -> 10 输出层
        )
        self.init_weights()
    
    def init_weights(self):
        """初始化网络权重
        
        使用Kaiming初始化权重，适合ReLU激活函数
        偏置项初始化为0
        """
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
        self.net.apply(init_weights)
    
    def forward(self, X):
        """前向传播
        
        Args:
            X: 输入张量，形状为(batch_size, 1, 28, 28)
        
        Returns:
            输出张量，形状为(batch_size, 10)
        """
        return self.net(X)

# ===================== 损失函数定义 =====================
def get_loss_fn():
    """返回用于训练的损失函数
    
    Returns:
        nn.CrossEntropyLoss: 交叉熵损失函数
        特点：
        1. 包含了Softmax操作
        2. 适用于多分类问题
        3. 输入是未归一化的预测值（logits）
    """
    return nn.CrossEntropyLoss()

# ===================== 训练函数 =====================
def train_model(net, train_iter, test_iter, num_epochs, lr):
    """训练模型
    
    Args:
        net (nn.Module): 神经网络模型
        train_iter (DataLoader): 训练数据加载器
        test_iter (DataLoader): 测试数据加载器
        num_epochs (int): 训练轮数
        lr (float): 学习率
    
    Returns:
        tuple: (训练损失列表, 训练准确率列表, 测试准确率列表)
    """
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 将模型移动到设备
    net = net.to(device)
    
    # 获取损失函数
    loss_fn = get_loss_fn()
    # 使用AdamW优化器，添加权重衰减
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(num_epochs):
        net.train()
        metric_train = d2l.Accumulator(3)
        for X, y in train_iter:
            # 将数据移动到设备
            X, y = X.to(device), y.to(device)
            
            trainer.zero_grad()
            y_hat = net(X)
            l = loss_fn(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric_train.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
        
        train_loss = metric_train[0] / metric_train[2]
        train_acc = metric_train[1] / metric_train[2]
        test_acc = evaluate_accuracy(net, test_iter, device)
        
        # 更新学习率
        scheduler.step(test_acc)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch + 1}: Train Loss {train_loss:.3f}, '
              f'Train Acc {train_acc:.3f}, Test Acc {test_acc:.3f}')
    
    return train_losses, train_accs, test_accs

# ===================== 评估准确率函数 =====================
def evaluate_accuracy(net, data_iter, device=None):
    """评估模型在数据集上的准确率
    
    Args:
        net (nn.Module): 神经网络模型
        data_iter (DataLoader): 数据加载器
        device (torch.device): 计算设备
        
    Returns:
        float: 准确率
    """
    if device is None:
        device = get_device()
        
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
    
    metric = d2l.Accumulator(2)  # 正确预测数、预测总数
    
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# ===================== 评估函数 =====================
def evaluate_model(net, test_iter, n_samples):
    """评估模型并展示预测结果
    
    Args:
        net (nn.Module): 神经网络模型
        test_iter (DataLoader): 测试数据加载器
        n_samples (int): 要展示的样本数量
    
    Returns:
        tuple: (预测结果, 真实标签)
    """
    device = get_device()
    net.eval()
    X_batch, y_batch_true = next(iter(test_iter))
    
    # 将数据移动到设备上进行预测
    X_batch = X_batch.to(device)
    
    with torch.no_grad():
        y_batch_pred_logits = net(X_batch)
        y_batch_pred_indices = y_batch_pred_logits.argmax(axis=1)
    
    # 将预测结果移回CPU以便显示
    y_batch_pred_indices = y_batch_pred_indices.cpu()
    
    return y_batch_pred_indices, y_batch_true 