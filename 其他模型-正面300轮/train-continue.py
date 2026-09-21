# from ultralytics import YOLO

# # Load a model
# model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML
# model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights

# # Train the model
# results = model.train(data="mnist160", epochs=100, imgsz=64)

import glob
import os

import pandas as pd
import torch

# 导入自定义-RegressionModel
from my_yolo_r_p1_c_s_att_conv.custom_modules.custom_tasks import RegressionModel
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

# 配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# MODEL_YAML_PATH = r'C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\my_yolo_r_p1_c_s_att_conv\yoloV11n-r-att-conv.yaml'
# PRETRAINED_WEIGHTS_PATH = 'yolo11n-cls.pt'  # 你的预训练权重
TRAIN_CSV_PATH = os.path.join(PROJECT_ROOT, "yolo8-12-正面300轮", "datasets", "train.csv")
VAL_CSV_PATH = os.path.join(PROJECT_ROOT, "yolo8-12-正面300轮", "datasets", "val.csv")
# SAVE_DIR = os.path.join('my_yolo_r_p1_c_s_att_conv', 'runs-yolo11n-AFAR')  # 可自定义

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 300  # 调整训练轮数看R2能否有所提高
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
IMG_SIZE = 224

# os.makedirs(SAVE_DIR, exist_ok=True)


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


def train_one_yaml(yaml_path):
    print(f"\n========== 开始训练: {yaml_path} ==========")
    # 保存路径改为yolo8-12-正面/runs-xxx
    yaml_name = os.path.splitext(os.path.basename(yaml_path))[0]
    save_dir = os.path.join(PROJECT_ROOT, "yolo8-12正面-300轮", f"runs-{yaml_name}")
    os.makedirs(save_dir, exist_ok=True)

    # 其余配置和数据加载不变
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

    net = RegressionModel(yaml_path, ch=3).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float("inf")
    train_losses, val_losses, val_maes, val_r2s = [], [], [], []

    for epoch in range(EPOCHS):
        net.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"[{yaml_name}] Epoch {epoch + 1}/{EPOCHS} [训练]")
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
        train_losses.append(avg_train_loss)

        net.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"[{yaml_name}] Epoch {epoch + 1}/{EPOCHS} [验证]")
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
        val_mae = mean_absolute_error(all_labels, all_preds)
        val_r2 = r2_score(all_labels, all_preds)
        val_maes.append(val_mae)
        val_r2s.append(val_r2)

        print(
            f"[{yaml_name}] Epoch {epoch + 1}/{EPOCHS} -> 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f} | 验证MAE: {val_mae:.4f} | 验证R2: {val_r2:.4f}"
        )
        scheduler.step()
        torch.save(net.state_dict(), os.path.join(save_dir, "last.pt"))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), os.path.join(save_dir, "best.pt"))
            print(f"🎉 新的最佳模型已保存，验证损失: {best_val_loss:.4f}")

    # 保存loss等
    with open(os.path.join(save_dir, "train_losses.txt"), "w", encoding="utf-8") as f:
        for loss in train_losses:
            f.write(f"{loss}\n")
    with open(os.path.join(save_dir, "val_losses.txt"), "w", encoding="utf-8") as f:
        for loss in val_losses:
            f.write(f"{loss}\n")
    with open(os.path.join(save_dir, "val_maes.txt"), "w", encoding="utf-8") as f:
        f.writelines(f"{mae}\n" for mae in val_maes)
    with open(os.path.join(save_dir, "val_r2s.txt"), "w", encoding="utf-8") as f:
        f.writelines(f"{r2}\n" for r2 in val_r2s)


if __name__ == "__main__":
    yaml_dir = os.path.join(PROJECT_ROOT, "yolo8-12-正面", "yaml")
    yaml_files = glob.glob(os.path.join(yaml_dir, "*.yaml"))
    print(f"共检测到 {len(yaml_files)} 个yaml文件，将依次训练：")
    for yaml_path in yaml_files:
        train_one_yaml(yaml_path)
