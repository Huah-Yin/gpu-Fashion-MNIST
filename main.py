from d2l import torch as d2l
import torch
from torch.utils import data
import time  # 添加time模块导入

from data_utils import load_fashion_mnist, get_fashion_mnist_labels
from model import FashionMNISTNet, train_model, evaluate_model, get_device
from visualization import show_images, plot_training_process

def main():
    # 记录开始时间
    start_time = time.time()
    
    device = get_device()
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU内存使用: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
        print(f"GPU内存峰值: {torch.cuda.max_memory_allocated(0) / 1024 ** 2:.2f} MB")
    
    # 训练参数
    lr = 0.001          # 学习率：控制模型参数更新的步长
    num_epochs = 30     # 训练轮数：完整遍历数据集的次数
    batch_size = 256    # 批次大小：每次训练使用的样本数量
    num_workers = 4     # 数据加载线程数：并行加载数据的线程数量

    # 网络结构参数
    num_inputs = 784    # 输入维度：28*28=784，表示每张图片的像素数
    num_hiddens = 256   # 隐藏层神经元数量：可以调整以改变模型复杂度
    num_outputs = 10    # 输出维度：10个类别（不同类型的衣物）

    # 评估参数
    num_samples = 6     # 评估时展示的预测结果数量（<= 10 只展示部分预测结果）

    # ===================== 程序执行 =====================
    # 设置显示
    d2l.use_svg_display()
    
    # 加载数据
    train_iter, test_iter = load_fashion_mnist(batch_size, num_workers=num_workers)
    
    # 显示数据形状信息
    X, y = next(iter(train_iter))
    print(f"Shape of the first training image: {X[0].shape}")
    
    # 显示初始训练样本（各种衣物类别）
    print("\n显示初始训练样本（各种衣物类别）：")
    X, y = next(iter(data.DataLoader(train_iter.dataset, batch_size=18)))
    show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
    
    # 创建和训练模型
    net = FashionMNISTNet(num_inputs, num_hiddens, num_outputs)
    print(f"\nStarting training for {num_epochs} epochs with learning rate : {lr}")
    # 绘制训练过程
    print("\n显示训练过程：")
    train_losses, train_accs, test_accs = train_model(net, train_iter, test_iter, num_epochs, lr)
    
    
    plot_training_process(train_losses, train_accs, test_accs, num_epochs)
    
    # 评估模型并显示预测结果
    y_pred, y_true = evaluate_model(net, test_iter, num_samples)
    true_labels = get_fashion_mnist_labels(y_true)
    pred_labels = get_fashion_mnist_labels(y_pred)
    
    # 显示预测结果
    print("\n显示最终预测结果：")
    titles = [f'True: {true}\nPred: {pred}' 
             for true, pred in zip(true_labels, pred_labels)]
    X_batch, _ = next(iter(data.DataLoader(test_iter.dataset, batch_size=num_samples)))
    show_images(X_batch, 1, num_samples, titles=titles, scale=2.0)
    
    # 打印最终评估结果
    print("\nFinal Evaluation:")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Training Accuracy: {train_accs[-1]:.3f}")
    print(f"Final Test Accuracy: {test_accs[-1]:.3f}")
    
    # 打印预测结果比较
    print("\nTextual comparison for the displayed batch:")
    for i in range(num_samples):
        print(f"Image {i+1}: True Label = {true_labels[i]}, Predicted Label = {pred_labels[i]}")
    
    # 计算并显示总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print(f"\n程序总运行时间: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
    
    # 如果使用GPU，显示最终内存使用情况
    if device.type == 'cuda':
        print(f"GPU最终内存使用: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
        print(f"GPU内存峰值: {torch.cuda.max_memory_allocated(0) / 1024 ** 2:.2f} MB")
        # 清理GPU内存
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 