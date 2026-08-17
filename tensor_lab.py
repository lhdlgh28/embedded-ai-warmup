import torch
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
x = torch.arange(32*128).float()
x = x.reshape(32,128)
w = torch.randn(128,256)
y = x @ w
print(x.shape)
print(w.shape)
print(y.shape)
bias = torch.randn(256)
z = y + bias
print(z.shape)
x = x.to(device)
w = w.to(device)
bias = bias.to(device)
z_gpu = x @ w + bias
print(z_gpu.device)
print(torch.mean(z_gpu))
print(torch.max(z_gpu))