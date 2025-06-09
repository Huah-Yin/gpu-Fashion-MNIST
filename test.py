import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU 索引: {torch.cuda.current_device()}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    # 尝试将一个张量移动到 GPU
    try:
        x = torch.rand(3, 3).cuda()
        print(f"成功将张量移动到 GPU，其设备为: {x.device}")
    except Exception as e:
        print(f"未能将张量移动到 GPU: {e}")
        print("这可能意味着 CUDA 运行时环境有问题。")
else:
    print("GPU 不可用，请检查你的 PyTorch 和 CUDA 安装。")

print("\n-------------------------------------------------")
print("如果上面显示 GPU 不可用或出错，请检查以下文件大小：")
print("/home/yjb/anaconda3/envs/gpu-d2l/lib/libcublasLt.so.12.1.0.26")
import os
file_path = "/home/yjb/anaconda3/envs/gpu-d2l/lib/libcublasLt.so.12.1.0.26"
if os.path.exists(file_path):
    file_size = os.path.getsize(file_path)
    print(f"文件存在，大小: {file_size} 字节")
    if file_size == 0:
        print("!!! 警告：文件大小为 0 字节，这是问题的根本原因。")
else:
    print("!!! 警告：文件不存在。")
