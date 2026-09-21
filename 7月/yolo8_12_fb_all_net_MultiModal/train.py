# train.py

import os

import pandas as pd
import torch

# ----------------- 1. 从自定义模块导入所需类 -----------------
# 确保可以从我们创建的模块中导入 RegressionModel
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

# 从 ultralytics 导入 YOLO 以便加载模型结构和权重
from ultralytics import YOLO

# ----------------- 2. 定义超参数和配置 -----------------
# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_YAML_PATH = os.path.join(PROJECT_ROOT, "yolov8n-regression.yaml")
PRETRAINED_WEIGHTS_PATH = "yolov8n.pt"  # 确保此文件已下载或存在
TRAIN_CSV_PATH = os.path.join(PROJECT_ROOT, "datasets", "train.csv")
VAL_CSV_PATH = os.path.join(PROJECT_ROOT, "datasets", "val.csv")
SAVE_DIR = os.path.join(PROJECT_ROOT, "runs")  # 保存模型权重和结果的目录

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
        img_abs_path = os.path.join(PROJECT_ROOT, img_relative_path)
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
    # 1. 使用YOLO类从.yaml文件构建模型结构
    # 2. .load()方法加载预训练的权重到这个结构中
    # 3. .model 提取出底层的PyTorch nn.Module
    model_wrapper = YOLO(MODEL_YAML_PATH).load(PRETRAINED_WEIGHTS_PATH)
    net = model_wrapper.model.to(DEVICE)
    print("模型加载并移动到设备。")

    # --- 损失函数和优化器 ---
    criterion = nn.MSELoss()  # 均方误差损失，回归任务首选
    # criterion = nn.L1Loss() # 也可使用L1损失 (平均绝对误差)
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 每10个epoch学习率乘以0.5

    # --- 训练循环 ---
    best_val_loss = float("inf")

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

        # 验证阶段
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [验证]")
            for images, labels in val_pbar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_pbar.set_postfix({"val_loss": loss.item()})

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{EPOCHS} -> 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f}")

        # 更新学习率
        scheduler.step()

        # --- 保存模型 ---
        # 保存最新的模型
        torch.save(net.state_dict(), os.path.join(SAVE_DIR, "last.pt"))

        # 如果验证损失创新低，则保存为最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), os.path.join(SAVE_DIR, "best.pt"))
            print(f"🎉 新的最佳模型已保存，验证损失: {best_val_loss:.4f}")

    print("\n训练完成！")
    print(f"最佳模型权重已保存到: {os.path.join(SAVE_DIR, 'best.pt')}")


if __name__ == "__main__":
    main()
