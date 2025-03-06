import torch

# 检查PyTorch版本
print(torch.__version__)

# 检查CUDA是否可用
print(torch.cuda.is_available())

# 查看可用的CUDA设备数量
print(torch.cuda.device_count())

# 查看PyTorch对应的CUDA版本
print(torch.version.cuda)