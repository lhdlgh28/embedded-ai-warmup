import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from models import MLP, CNNSmall, CNNLarge


# ==========================================
# 1. 基本配置
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


models = {
    "MLP": MLP(),
    "CNN-Small": CNNSmall(),
    "CNN-Large": CNNLarge()
}


checkpoint_paths = {
    "MLP": "checkpoints/mlp.pth",
    "CNN-Small": "checkpoints/cnn_small.pth",
    "CNN-Large": "checkpoints/cnn_large.pth"
}


# 已经得到的测试集 Accuracy
accuracies = {
    "MLP": 82.22,
    "CNN-Small": 84.93,
    "CNN-Large": 79.59
}


# ==========================================
# 2. Latency benchmark 函数
# ==========================================

def benchmark_latency(
    model,
    device,
    warmup=20,
    runs=200
):

    model = model.to(device)
    model.eval()

    # batch size = 1
    x = torch.randn(
        1,
        1,
        28,
        28,
        device=device
    )

    # -------------------------
    # Warm-up
    # -------------------------

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()


    # -------------------------
    # 正式测量
    # -------------------------

    times = []

    with torch.no_grad():

        for _ in range(runs):

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            _ = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            latency_ms = (end - start) * 1000

            times.append(latency_ms)


    # -------------------------
    # 统计
    # -------------------------

    mean_latency = np.mean(times)

    median_latency = np.median(times)

    p95_latency = np.percentile(
        times,
        95
    )

    return (
        mean_latency,
        median_latency,
        p95_latency
    )


# ==========================================
# 3. 创建 results 文件夹
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)


# ==========================================
# 4. Benchmark 三个模型
# ==========================================

results = []


for name, model in models.items():

    print(f"\n===== {name} =====")

    # -------------------------
    # 参数量
    # -------------------------

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: {num_params:,}"
    )


    # -------------------------
    # 模型文件大小
    # -------------------------

    checkpoint_path = checkpoint_paths[name]

    size_bytes = os.path.getsize(
        checkpoint_path
    )

    size_mb = (
        size_bytes /
        (1024 ** 2)
    )

    print(
        f"Checkpoint Size: "
        f"{size_mb:.4f} MB"
    )


    # -------------------------
    # 加载训练好的权重
    # -------------------------

    state_dict = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )


    # -------------------------
    # Latency
    # -------------------------

    mean, median, p95 = benchmark_latency(
        model,
        device,
        warmup=20,
        runs=200
    )

    print(
        f"Mean: {mean:.4f} ms"
    )

    print(
        f"Median: {median:.4f} ms"
    )

    print(
        f"P95: {p95:.4f} ms"
    )


    # -------------------------
    # 保存到 results
    # -------------------------

    results.append({
        "Model": name,
        "Accuracy (%)": accuracies[name],
        "Parameters": num_params,
        "Size (MB)": size_mb,
        "Mean Latency (ms)": mean,
        "Median Latency (ms)": median,
        "P95 Latency (ms)": p95
    })


# ==========================================
# 5. 转成 DataFrame
# ==========================================

df = pd.DataFrame(
    results
)


print("\n===== Final Results =====")

print(df)


# ==========================================
# 6. 保存 results.csv
# ==========================================

csv_path = "results/results.csv"

df.to_csv(
    csv_path,
    index=False
)

print(
    f"\nResults saved to: "
    f"{csv_path}"
)


# ==========================================
# 7. 绘制 benchmark 图
# ==========================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8)
)


# --------------------------
# Accuracy
# --------------------------

axes[0, 0].bar(
    df["Model"],
    df["Accuracy (%)"]
)

axes[0, 0].set_title(
    "Test Accuracy"
)

axes[0, 0].set_ylabel(
    "Accuracy (%)"
)


# --------------------------
# Parameters
# --------------------------

axes[0, 1].bar(
    df["Model"],
    df["Parameters"]
)

axes[0, 1].set_title(
    "Parameter Count"
)

axes[0, 1].set_ylabel(
    "Parameters"
)


# --------------------------
# Model Size
# --------------------------

axes[1, 0].bar(
    df["Model"],
    df["Size (MB)"]
)

axes[1, 0].set_title(
    "Checkpoint Size"
)

axes[1, 0].set_ylabel(
    "Size (MB)"
)


# --------------------------
# Latency
# --------------------------

x = np.arange(
    len(df)
)

width = 0.25

axes[1, 1].bar(
    x - width,
    df["Mean Latency (ms)"],
    width,
    label="Mean"
)

axes[1, 1].bar(
    x,
    df["Median Latency (ms)"],
    width,
    label="Median"
)

axes[1, 1].bar(
    x + width,
    df["P95 Latency (ms)"],
    width,
    label="P95"
)

axes[1, 1].set_xticks(x)

axes[1, 1].set_xticklabels(
    df["Model"]
)

axes[1, 1].set_title(
    "Inference Latency"
)

axes[1, 1].set_ylabel(
    "Latency (ms)"
)

axes[1, 1].legend()


# --------------------------
# 保存图片
# --------------------------

plt.tight_layout()

figure_path = (
    "results/benchmark.png"
)

plt.savefig(
    figure_path,
    dpi=300
)

plt.close()


print(
    f"Benchmark figure saved to: "
    f"{figure_path}"
)