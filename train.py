import os
import pandas as pd

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# =========================
# 1. Dataset / DataLoader
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

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# =========================
# 2. 定义模型
# =========================
from models import MLP, CNNSmall, CNNLarge

# =========================
# 3. Device
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device:", device)


# =========================
# 4. 创建 CNN 模型
# =========================

model = CNNLarge().to(device)


# =========================
# 5. Loss
# =========================

criterion = nn.CrossEntropyLoss()


# =========================
# 6. Optimizer
# =========================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


# =========================
# 7. 保存路径
# =========================

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

checkpoint_path = "checkpoints/cnn_large_checkpoint.pth"
model_path = "checkpoints/cnn_large.pth"
history_path = "results/cnn_large_history.csv"


# =========================
# 8. 尝试恢复 checkpoint
# =========================

num_epochs = 5
start_epoch = 0
history = []

if os.path.exists(checkpoint_path):

    print("CNN checkpoint found. Loading...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"]

    history = checkpoint.get("history", [])

    print(
        f"Checkpoint loaded. "
        f"Resume from epoch {start_epoch + 1}."
    )


# =========================
# 9. Training
# =========================

for epoch in range(start_epoch, num_epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    history.append({
        "epoch": epoch + 1,
        "loss": average_loss
    })

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {average_loss:.4f}"
    )

    # 每个 epoch 保存 CNN checkpoint
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": average_loss,
            "history": history
        },
        checkpoint_path
    )

    # 保存训练历史
    df = pd.DataFrame(history)

    df.to_csv(
        history_path,
        index=False
    )

    print("CNN checkpoint saved.")


# =========================
# 10. Test
# =========================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


accuracy = correct / total

print(
    f"Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# =========================
# 11. 保存最终 CNN 模型
# =========================

torch.save(
    model.state_dict(),
    model_path
)

print("CNN final model saved.")


# =========================
# 12. 保存最终历史
# =========================

df = pd.DataFrame(history)

df.to_csv(
    history_path,
    index=False
)

print("CNN training history saved.")