import torch
import torchvision
from torch.utils import data
from torchvision import transforms

def load_fashion_mnist(batch_size=256, root="./data", num_workers=4):
    """加载Fashion-MNIST数据集
    
    Args:
        batch_size (int): 批次大小
        root (str): 数据存储路径，默认为当前目录下的data文件夹
        num_workers (int): 数据加载线程数，默认为4
    
    Returns:
        tuple: (训练数据加载器, 测试数据加载器)
    """
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
    # 并除以255使得所有像素的数值均在0～1之间
    # ToTensor() 转换会：
    # 1. 将PIL图像或numpy数组转换为张量
    # 2. 将像素值从[0, 255]缩放到[0.0, 1.0]范围
    # 3. 将图像格式从(H, W, C)转换为(C, H, W)格式
    trans = transforms.ToTensor()
    try:
        mnist_train = torchvision.datasets.FashionMNIST(
            root=root, train=True, transform=trans, download=False)
        mnist_test = torchvision.datasets.FashionMNIST(
            root=root, train=False, transform=trans, download=False)
    except Exception as e:
        print(f"Error loading FashionMNIST dataset: {e}")
        print("Please check if the dataset exists in the specified path.")
        raise

    # 检测是否可以使用GPU，如果可以则使用pin_memory加速数据传输
    pin_memory = torch.cuda.is_available()
    
    # 使用多线程加载数据以提高效率
    train_iter = data.DataLoader(
        mnist_train, 
        batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    test_iter = data.DataLoader(
        mnist_test, 
        batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    return train_iter, test_iter

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签
    
    Args:
        labels (torch.Tensor): 标签张量
    
    Returns:
        list: 文本标签列表
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels] 