import os

import matplotlib.pyplot as plt

# 文件路径
train_loss_file = (
    r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\my_yolo_regression_project1\runs\train_losses.txt"
)
val_loss_file = (
    r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\my_yolo_regression_project1\runs\val_losses.txt"
)
val_mae_file = (
    r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\my_yolo_regression_project1\runs\val_maes.txt"
)
val_r2_file = (
    r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\my_yolo_regression_project1\runs\val_r2s.txt"
)


def read_txt(file):
    with open(file, encoding="utf-8") as f:
        lines = f.readlines()
    # 跳过REM注释行，只保留数字
    return [float(line.strip()) for line in lines if not line.strip().startswith("REM") and line.strip()]


train_losses = read_txt(train_loss_file)
val_losses = read_txt(val_loss_file)
val_maes = read_txt(val_mae_file)
val_r2s = read_txt(val_r2_file)

epochs = range(1, len(train_losses) + 1)

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

save_dir = "runs"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "metrics_curve.png"), dpi=200)  # 保存图片
plt.show()
