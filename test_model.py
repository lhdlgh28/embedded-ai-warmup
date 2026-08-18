from models import MLP
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import MLP


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

transform = transforms.ToTensor()

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


model = MLP().to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/mlp.pth",
        map_location=device
    )
)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total

print(f"Loaded Model Accuracy: {accuracy * 100:.2f}%")