import os

import matplotlib.pyplot as plt


def read_txt(file):
    with open(file, encoding="utf-8") as f:
        lines = f.readlines()
    return [float(line.strip()) for line in lines if not line.strip().startswith("REM") and line.strip()]


# 获取当前脚本所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# 遍历所有以runs开头的文件夹
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path) and folder.startswith("runs"):
        print(f"正在处理: {folder_path}")

        # 四个指标文件路径
        train_loss_file = os.path.join(folder_path, "train_losses.txt")
        val_loss_file = os.path.join(folder_path, "val_losses.txt")
        val_mae_file = os.path.join(folder_path, "val_maes.txt")
        val_r2_file = os.path.join(folder_path, "val_r2s.txt")

        # 检查文件是否都存在
        if not all(os.path.exists(f) for f in [train_loss_file, val_loss_file, val_mae_file, val_r2_file]):
            print(f"跳过 {folder}，缺少部分结果文件。")
            continue

        # 读取数据
        train_losses = read_txt(train_loss_file)
        val_losses = read_txt(val_loss_file)
        val_maes = read_txt(val_mae_file)
        val_r2s = read_txt(val_r2_file)
        epochs = range(1, len(train_losses) + 1)

        # 绘图
        plt.figure(figsize=(14, 8))
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train Loss")
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(epochs, val_losses, label="Val Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(epochs, val_maes, label="Val MAE", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("Validation MAE")
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(epochs, val_r2s, label="Val R2", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("R2 Score")
        plt.title("Validation R2")
        plt.grid(True)

        plt.tight_layout()
        # 保存到对应文件夹
        save_path = os.path.join(folder_path, "metrics_curve.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"已保存: {save_path}")
