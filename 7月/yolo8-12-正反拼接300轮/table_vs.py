import csv
import os


def read_txt(file):
    with open(file, encoding="utf-8") as f:
        lines = f.readlines()
    return [float(line.strip()) for line in lines if not line.strip().startswith("REM") and line.strip()]


# 获取当前脚本所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# 用于保存所有实验的最优指标
results = []

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

        # 统计最优指标
        min_train_loss = min(train_losses)
        min_val_loss = min(val_losses)
        min_val_mae = min(val_maes)
        max_val_r2 = max(val_r2s)

        # 记录到结果列表
        results.append([folder, min_train_loss, min_val_loss, min_val_mae, max_val_r2])

# 写入csv文件
csv_path = os.path.join(base_dir, "all_best_metrics.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["folder", "min_train_loss", "min_val_loss", "min_val_mae", "max_val_r2"])
    writer.writerows(results)

print(f"所有实验最优指标已保存到: {csv_path}")
