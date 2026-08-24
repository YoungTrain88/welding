# train.py

import os

import pandas as pd
import torch

# ----------------- 1. 从自定义模块导入所需类 -----------------
# 确保可以从我们创建的模块中导入 RegressionModel
from my_yolo_regression_project1.custom_modules.custom_tasks import RegressionModel
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

# 从 ultralytics 导入 YOLO 以便加载模型结构和权重

# ----------------- 2. 定义超参数和配置 -----------------
# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_YAML_PATH = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\my_yolo_regression_project1-cat-shuffed\yolov8n-regression.yaml"
PRETRAINED_WEIGHTS_PATH = "yolov8n.pt"  # 确保此文件已下载或存在
TRAIN_CSV_PATH = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\my_yolo_regression_project1-cat-shuffed\datasets\train.csv"
VAL_CSV_PATH = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\my_yolo_regression_project1-cat-shuffed\datasets\val.csv"
SAVE_DIR = os.path.join(r"my_yolo_regression_project1-cat-shuffed", "runs-yolov850")  # 保存模型权重和结果的目录

# 训练配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
IMG_SIZE = 224

# 确保保存目录存在
os.makedirs(SAVE_DIR, exist_ok=True)


# ----------------- 3. 自定义数据集类 -----------------
class RegressionDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data_frame = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_relative_path = self.data_frame.iloc[idx, 0]
        img_abs_path = os.path.join(PROJECT_ROOT, "my_yolo_regression_project1-cat-shuffed", img_relative_path)
        image = Image.open(img_abs_path).convert("RGB")
        value = torch.tensor([self.data_frame.iloc[idx, 1]], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, value


# ----------------- 4. 主训练逻辑 -----------------
def main():
    print(f"使用设备: {DEVICE}")

    # --- 数据加载 ---
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = RegressionDataset(csv_path=TRAIN_CSV_PATH, transform=transform)
    val_dataset = RegressionDataset(csv_path=VAL_CSV_PATH, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("数据加载器准备完毕。")

    # --- 模型初始化 ---
    # ----------------------------- 新的修改部分开始 -----------------------------

    # 1. 直接实例化我们自己的 RegressionModel
    #    我们不再使用 YOLO() 封装类。这里的 ch=3 表示输入通道为3 (RGB图片)
    print("直接从 YAML 创建自定义 RegressionModel...")
    net = RegressionModel(MODEL_YAML_PATH, ch=3).to(DEVICE)

    # 2. 加载预训练权重
    print(f"加载预训练权重从: {PRETRAINED_WEIGHTS_PATH}")

    # 加载预训练的 .pt 文件，它是一个包含 state_dict 的字典
    ckpt = torch.load(PRETRAINED_WEIGHTS_PATH, map_location=DEVICE)

    # 将 state_dict 加载到我们的模型中
    # 使用 strict=False 是【关键】
    # 这允许我们只加载那些在两个模型中名称和尺寸都匹配的层（即骨干网络部分），
    # 而忽略不匹配的层（例如我们自己的 RegressionHead 和原来的 Detect Head）。
    state_dict = ckpt["model"].float().state_dict()
    net.load_state_dict(state_dict, strict=False)

    print("模型加载并移动到设备。")
    # --- 损失函数和优化器 ---
    criterion = nn.MSELoss()  # 均方误差损失，回归任务首选
    # criterion = nn.L1Loss() # 也可使用L1损失 (平均绝对误差)
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 每10个epoch学习率乘以0.5

    # --- 训练循环 ---
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    val_maes = []
    val_r2s = []

    for epoch in range(EPOCHS):
        # 训练阶段
        net.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [训练]")

        for images, labels in train_pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)  # 记录训练损失

        # 验证阶段
        net.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [验证]")
            for images, labels in val_pbar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_pbar.set_postfix({"val_loss": loss.item()})
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 计算回归指标
        val_mae = mean_absolute_error(all_labels, all_preds)
        val_r2 = r2_score(all_labels, all_preds)
        val_maes.append(val_mae)
        val_r2s.append(val_r2)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} -> 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f} | 验证MAE: {val_mae:.4f} | 验证R2: {val_r2:.4f}"
        )

        # 更新学习率
        scheduler.step()

        # --- 保存模型 ---
        torch.save(net.state_dict(), os.path.join(SAVE_DIR, "last.pt"))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), os.path.join(SAVE_DIR, "best.pt"))
            print(f"🎉 新的最佳模型已保存，验证损失: {best_val_loss:.4f}")

    print("\n训练完成！")
    print(f"最佳模型权重已保存到: {os.path.join(SAVE_DIR, 'best.pt')}")
    # 保存损失曲线为txt格式
    with open(os.path.join(SAVE_DIR, "train_losses.txt"), "w", encoding="utf-8") as f:
        for loss in train_losses:
            f.write(f"{loss}\n")
    with open(os.path.join(SAVE_DIR, "val_losses.txt"), "w", encoding="utf-8") as f:
        for loss in val_losses:
            f.write(f"{loss}\n")
    # 保存指标
    with open(os.path.join(SAVE_DIR, "val_maes.txt"), "w", encoding="utf-8") as f:
        f.writelines(f"{mae}\n" for mae in val_maes)
    with open(os.path.join(SAVE_DIR, "val_r2s.txt"), "w", encoding="utf-8") as f:
        f.writelines(f"{r2}\n" for r2 in val_r2s)


if __name__ == "__main__":
    main()
