import torch
print(torch.cuda.is_available())  
print("alright")
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")

