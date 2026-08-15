import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))