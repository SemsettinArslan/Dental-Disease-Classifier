import torch

print("====================================")
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Current Device Index:", torch.cuda.current_device())
    print("GPU Device Name:", torch.cuda.get_device_name(0))
    print("CUDA Device Capability:", torch.cuda.get_device_capability(0))
else:
    print("CUDA is NOT available in this PyTorch installation.")
print("====================================")
