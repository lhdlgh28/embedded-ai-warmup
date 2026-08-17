import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# =========================
# 1. 准备数据
# =========================

transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# =========================
# 2. 定义 MLP
# =========================

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


# =========================
# 3. 创建模型
# =========================

model = MLP()


# =========================
# 4. 从 DataLoader 拿一个 batch
# =========================

x, labels = next(iter(train_loader))

print("x shape:", x.shape)


# =========================
# 5. 前向传播
# =========================

output = model(x)

print("output shape:", output.shape)


# =========================
# 6. 找出预测类别
# =========================

predictions = output.argmax(dim=1)

print("predictions shape:", predictions.shape)
print("first 10 predictions:", predictions[:10])