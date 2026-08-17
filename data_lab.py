import torch
from torchvision import datasets, transforms

transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

print(len(train_dataset))
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
print("dataset size:", len(train_dataset))

image, label = train_dataset[0]

print("single image:", image.shape)
print("single label:", label)

images, labels = next(iter(train_loader))

print("batch images:", images.shape)
print("batch labels:", labels.shape)