import torch

# 训练数据
x = torch.tensor(2.0)
y = torch.tensor(10.0)

# 需要学习的参数
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

lr = 0.01

# 创建优化器
optimizer = torch.optim.SGD([w, b], lr=lr)

for epoch in range(20):

    # 1. prediction
    prediction = w * x + b

    # 2. loss
    loss = (prediction - y) ** 2

    # 3. backward
    loss.backward()

    # 4. 更新 w 和 b
    optimizer.step()

    # 5. 清空梯度
    optimizer.zero_grad()

    print(
        f"epoch={epoch + 1:2d}, "
        f"prediction={prediction.item():.4f}, "
        f"loss={loss.item():.4f}, "
        f"w={w.item():.4f}, "
        f"b={b.item():.4f}"
    )