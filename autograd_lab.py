import torch

# 训练数据
x = torch.tensor(2.0)
y = torch.tensor(10.0)

# 参数
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 学习率
lr = 0.01

for epoch in range(20):

    # 1. prediction
    prediction = w * x + b

    # 2. loss
    loss = (prediction - y) ** 2

    # 3. backward
    loss.backward()

    # 4. 更新参数
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # 5. 清空梯度
    w.grad.zero_()
    b.grad.zero_()

    # 打印
    print(
        f"epoch={epoch + 1:2d}, "
        f"prediction={prediction.item():.4f}, "
        f"loss={loss.item():.4f}, "
        f"w={w.item():.4f}, "
        f"b={b.item():.4f}"
    )