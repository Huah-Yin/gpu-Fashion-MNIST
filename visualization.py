import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表
    
    Args:
        imgs (torch.Tensor): 图像张量
        num_rows (int): 行数
        num_cols (int): 列数
        titles (list, optional): 标题列表
        scale (float): 图像缩放比例
    """
    plt.ioff()  # 关闭交互模式
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            if img.ndim == 3 and img.shape[0] == 1:
                img_display = img.squeeze().numpy()
            elif img.ndim == 2:
                img_display = img.numpy()
            else:
                img_display = img.numpy()
            ax.imshow(img_display)
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()  # 显示弹窗
    plt.close(fig)  # 关闭图形，防止输出Figure size信息
    return axes

def plot_training_process(train_losses, train_accs, test_accs, num_epochs):
    """绘制训练过程
    
    Args:
        train_losses (list): 训练损失列表
        train_accs (list): 训练准确率列表
        test_accs (list): 测试准确率列表
        num_epochs (int): 训练轮数
    """
    plt.ioff()  # 关闭交互模式
    fig, ax = plt.subplots(figsize=(8, 6))
    
    epochs = range(1, num_epochs + 1)
    ax.plot(epochs, train_losses, 'b-', label='train loss')
    ax.plot(epochs, train_accs, 'g-', label='train acc')
    ax.plot(epochs, test_accs, 'r-', label='test acc')
    
    ax.set_xlabel('epoch')
    ax.set_ylabel('value')
    ax.set_xlim([1, num_epochs])
    ax.set_ylim([0.0, 1.0])
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    plt.close(fig) 