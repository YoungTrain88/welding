# train_resnet.py

import os

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import transforms
from tqdm import tqdm

# ===================================================================
# 1. 定义超参数和配置
# ===================================================================

# 路径配置 (脚本会自动获取当前目录作为项目根目录)'
PROJECT_ROOT = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\my_yolo_regression_project1-cat-shuffed"
# 假设您的数据存储在项目根目录下的 'datasets' 文件夹中
TRAIN_CSV_PATH = os.path.join(PROJECT_ROOT, "datasets", "train.csv")
VAL_CSV_PATH = os.path.join(PROJECT_ROOT, "datasets", "val.csv")
SAVE_DIR = os.path.join(PROJECT_ROOT, "runs_resnet50")  # 为ResNet创建一个新的保存目录

# 训练配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 300
BATCH_SIZE = 16  # 如果显存不足，可以调低此值，例如 8
LEARNING_RATE = 1e-4
IMG_SIZE = 224  # ResNet通常使用 224x224 或 256x256 的输入尺寸

# 确保保存目录存在
os.makedirs(SAVE_DIR, exist_ok=True)


# ===================================================================
# 2. ResNet50 回归模型创建函数
# ===================================================================
def create_resnet50_regression():
    """加载预训练的 ResNet50 并将其末层修改为回归头。."""
    # 1. 加载预训练的 ResNet50 模型
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # 2. 获取原始分类头的输入特征数
    # 在 ResNet 中，分类头名为 'fc' (fully-connected)
    num_ftrs = model.fc.in_features

    # 3. 用一个新的线性层替换掉原来的分类头
    # 新的线性层输出维度为1，用于回归
    model.fc = nn.Linear(num_ftrs, 1)

    return model


# ===================================================================
# 3. 自定义数据集类 (与之前完全相同)
# ===================================================================
class RegressionDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data_frame = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 这里的路径是相对于项目根目录的，例如 'datasets/images/001.jpg'
        img_relative_path = self.data_frame.iloc[idx, 0]
        img_abs_path = os.path.join(PROJECT_ROOT, img_relative_path)

        try:
            image = Image.open(img_abs_path).convert("RGB")
        except FileNotFoundError:
            print(f"错误: 找不到图片 {img_abs_path}")
            # 返回一个假的空数据，或者您可以选择抛出异常
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), torch.tensor([0.0])

        value = torch.tensor([self.data_frame.iloc[idx, 1]], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, value


# ===================================================================
# 4. 主训练逻辑
# ===================================================================
def main():
    print(f"使用设备: {DEVICE}")
    print("-" * 30)

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("数据加载器准备完毕。")
    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")
    print("-" * 30)

    # --- 模型初始化 ---
    print("正在创建 ResNet50 回归模型...")
    net = create_resnet50_regression().to(DEVICE)
    print("模型加载并移动到设备。")
    print("-" * 30)

    # --- 损失函数和优化器 ---
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

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
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [训练中]")
        for images, labels in train_pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        net.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
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

    # 保存损失和指标
    with open(os.path.join(SAVE_DIR, "train_losses.txt"), "w", encoding="utf-8") as f:
        for loss in train_losses:
            f.write(f"{loss}\n")
    with open(os.path.join(SAVE_DIR, "val_losses.txt"), "w", encoding="utf-8") as f:
        for loss in val_losses:
            f.write(f"{loss}\n")
    with open(os.path.join(SAVE_DIR, "val_maes.txt"), "w", encoding="utf-8") as f:
        f.writelines(f"{mae}\n" for mae in val_maes)
    with open(os.path.join(SAVE_DIR, "val_r2s.txt"), "w", encoding="utf-8") as f:
        f.writelines(f"{r2}\n" for r2 in val_r2s)


if __name__ == "__main__":
    # 确保您的CSV文件和图片路径正确
    if not os.path.exists(TRAIN_CSV_PATH) or not os.path.exists(VAL_CSV_PATH):
        print("=" * 50)
        print("错误：找不到训练或验证CSV文件。")
        print(f"请确保 '{TRAIN_CSV_PATH}' 和 '{VAL_CSV_PATH}' 文件存在。")
        print("=" * 50)
    else:
        main()
