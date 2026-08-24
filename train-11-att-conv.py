# from ultralytics import YOLO

# # Load a model
# model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML
# model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights

# # Train the model
# results = model.train(data="mnist160", epochs=100, imgsz=64)

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

PROJECT_NAME = "yolo8-12-正面"
YAML_NAME = "yolov8n"  # 你的YAML配置文件
EPOCHS = 100
# 配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_YAML_PATH = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\yolo8-12-正面\yaml\yolov8n.yaml"
PRETRAINED_WEIGHTS_PATH = "yolo11n-cls.pt"  # 你的预训练权重
TRAIN_CSV_PATH = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\yolo8-12-正面\datasets\train.csv"
VAL_CSV_PATH = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\yolo8-12-正面\datasets\val.csv"
SAVE_DIR = os.path.join(PROJECT_NAME, "runs-" + YAML_NAME)  # 可自定义

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
IMG_SIZE = 224

os.makedirs(SAVE_DIR, exist_ok=True)


class RegressionDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data_frame = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_relative_path = self.data_frame.iloc[idx, 0]
        img_abs_path = os.path.join(PROJECT_ROOT, PROJECT_NAME, img_relative_path)
        image = Image.open(img_abs_path).convert("RGB")
        value = torch.tensor([self.data_frame.iloc[idx, 1]], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, value


def main():
    print(f"使用设备: {DEVICE}")

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

    print("直接从 YAML 创建自定义 RegressionModel...")
    net = RegressionModel(MODEL_YAML_PATH, ch=3).to(DEVICE)

    # print(f"加载预训练权重从: {PRETRAINED_WEIGHTS_PATH}")
    # ckpt = torch.load(PRETRAINED_WEIGHTS_PATH, map_location=DEVICE)
    # state_dict = ckpt['model'].float().state_dict()
    # net.load_state_dict(state_dict, strict=False)
    print("模型加载并移动到设备。")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float("inf")
    train_losses, val_losses, val_maes, val_r2s = [], [], [], []

    for epoch in range(EPOCHS):
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
        train_losses.append(avg_train_loss)

        net.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
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
        val_mae = mean_absolute_error(all_labels, all_preds)
        val_r2 = r2_score(all_labels, all_preds)
        val_maes.append(val_mae)
        val_r2s.append(val_r2)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} -> 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f} | 验证MAE: {val_mae:.4f} | 验证R2: {val_r2:.4f}"
        )
        scheduler.step()
        torch.save(net.state_dict(), os.path.join(SAVE_DIR, "last.pt"))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), os.path.join(SAVE_DIR, "best.pt"))
            print(f"🎉 新的最佳模型已保存，验证损失: {best_val_loss:.4f}")

    print("\n训练完成！")
    print(f"最佳模型权重已保存到: {os.path.join(SAVE_DIR, 'best.pt')}")
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
    main()
