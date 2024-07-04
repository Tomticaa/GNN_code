import torch

print("PyTorch版本: ", torch.__version__)
print("CUDA是否可用: ", torch.cuda.is_available())
print("CUDA版本: ", torch.version.cuda)
print("cuDNN版本: ", torch.backends.cudnn.version())
print("CUDA设备数量: ", torch.cuda.device_count())
print("CUDA设备名称: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")
