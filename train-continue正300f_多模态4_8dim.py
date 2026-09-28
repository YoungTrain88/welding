# train_yolo_multimodal_custom_dims.py

import glob
import os

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from torchvision.transforms import transforms
from tqdm import tqdm

# 导入我们之前适配好的、可以返回特征的RegressionModel
from yolo8_12_f_all_net_MultiModal_dim.custom_modules.custom_tasks import RegressionModel

# ===================================================================
# 1. 配置 (与您提供的代码保持一致)
# ===================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# ##################################################################
# ## (显著标识) 请务必检查这里的 PROJECT_NAME 是否与您的文件夹名完全一致！##
# ## 根据您的报错信息，它应该是 'yolo8_12_f_all_net_MultiModal_dim'
# ##################################################################
PROJECT_NAME = "yolo8_12_f_all_net_MultiModal_dim"
EPOCHS = 150
TRAIN_CSV_PATH = os.path.join(PROJECT_ROOT, PROJECT_NAME, "datasets", "train.csv")
VAL_CSV_PATH = os.path.join(PROJECT_ROOT, PROJECT_NAME, "datasets", "val.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-2  # 保持正则化
IMG_SIZE = 224


# ===================================================================
# 2. 数据集类 (核心修正处)
# ===================================================================
class RegressionDataset(Dataset):
    def __init__(self, csv_path, project_root, project_name, transform=None):
        """初始化函数现在也接收 project_name，以确保路径拼接正确。."""
        self.data_frame = pd.read_csv(csv_path).dropna()
        self.transform = transform
        self.project_root = project_root
        self.project_name = project_name  # 存储项目名称
        self.target_values = self.data_frame.iloc[:, 1].values.astype("float32")
        self.tabular_data = self.data_frame.iloc[:, 2:].values.astype("float32")
        self.image_paths = self.data_frame.iloc[:, 0].values
        self.num_tabular_features = self.tabular_data.shape[1]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_relative_path = self.image_paths[idx].replace("\\", "/")

        # ##################################################################
        # ## (显著标识) 关键修正: 使用 project_name 来构建正确的绝对路径 ##
        # ##################################################################
        img_abs_path = os.path.join(self.project_root, self.project_name, img_relative_path)

        try:
            image = Image.open(img_abs_path).convert("RGB")
        except FileNotFoundError:
            error_msg = (
                f"\n\n[文件未找到错误!]\n"
                f"试图访问的图片路径: {img_abs_path}\n"
                f"这个路径由以下三部分拼接而成:\n"
                f"  1. 脚本根目录 (PROJECT_ROOT): {self.project_root}\n"
                f"  2. 项目文件夹名 (PROJECT_NAME): {self.project_name}\n"
                f"  3. 从CSV读到的相对路径: {img_relative_path}\n"
                f"请检查:\n"
                f"  - 您的图片是否真的存放在这个最终路径下?\n"
                f"  - 脚本顶部的 PROJECT_NAME 变量是否与您的文件夹名完全一致?\n"
            )
            raise FileNotFoundError(error_msg)

        tabular_features = torch.tensor(self.tabular_data[idx], dtype=torch.float32)
        target_value = torch.tensor([self.target_values[idx]], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, tabular_features, target_value


# ===================================================================
# 3. 多模态模型定义 (与之前版本一致)
# ===================================================================
class MultiModalModel(nn.Module):
    """一个可以自定义视觉和表格特征维度的多模态YOLO模型。."""

    def __init__(
        self, yaml_path, num_tabular_features, image_output_dim=8, tabular_output_dim=4, hidden_dim=64, dropout_rate=0.5
    ):
        super().__init__()

        self.image_branch = RegressionModel(yaml_path, ch=3, verbose=False)
        original_image_feature_dim = 1280

        self.image_feature_compressor = nn.Sequential(
            nn.Linear(original_image_feature_dim, image_output_dim), nn.ReLU()
        )

        self.tabular_branch = nn.Sequential(
            nn.Linear(num_tabular_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, tabular_output_dim),
        )

        fusion_input_dim = image_output_dim + tabular_output_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, 1)
        )

    def forward(self, image, tabular_data):
        original_image_features = self.image_branch(image, return_features=True)
        compressed_image_features = self.image_feature_compressor(original_image_features)
        tabular_features = self.tabular_branch(tabular_data)
        combined_features = torch.cat([compressed_image_features, tabular_features], dim=1)
        output = self.fusion_head(combined_features)
        return output


# ===================================================================
# 4. 训练函数 (已修正)
# ===================================================================
def train_one_yaml(yaml_path):
    print(f"\n========== 开始训练: {yaml_path} ==========")
    yaml_name = os.path.splitext(os.path.basename(yaml_path))[0]
    save_dir = os.path.join(PROJECT_ROOT, PROJECT_NAME, f"runs-{yaml_name}")
    os.makedirs(save_dir, exist_ok=True)

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # ##################################################################
    # ## (显著标识) 关键修正: 创建Dataset时，传入PROJECT_NAME ##
    # ##################################################################
    train_dataset = RegressionDataset(
        csv_path=TRAIN_CSV_PATH, project_root=PROJECT_ROOT, project_name=PROJECT_NAME, transform=train_transform
    )
    val_dataset = RegressionDataset(
        csv_path=VAL_CSV_PATH, project_root=PROJECT_ROOT, project_name=PROJECT_NAME, transform=val_transform
    )

    num_tabular_features = train_dataset.num_tabular_features

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"检测到 {num_tabular_features} 个表格特征。")

    net = MultiModalModel(
        yaml_path=yaml_path, num_tabular_features=num_tabular_features, image_output_dim=8, tabular_output_dim=4
    ).to(DEVICE)

    print("\n" + "=" * 50)
    print(f"             模型结构: {os.path.basename(yaml_path)}")
    print("=" * 50)
    summary(
        net,
        input_size=[(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE), (BATCH_SIZE, num_tabular_features)],
        depth=3,
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
    )
    print("=" * 50 + "\n")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.5, verbose=True)

    best_val_loss = float("inf")
    train_losses, val_losses, val_maes, val_r2s = [], [], [], []

    for epoch in range(EPOCHS):
        net.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"[{yaml_name}] Epoch {epoch + 1}/{EPOCHS} [训练]")
        for images, tabular, labels in train_pbar:
            images, tabular, labels = images.to(DEVICE), tabular.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images, tabular)
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
            for images, tabular, labels in val_pbar:
                images, tabular, labels = images.to(DEVICE), tabular.to(DEVICE), labels.to(DEVICE)
                outputs = net(images, tabular)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        avg_val_loss = val_loss / len(val_loader)
        val_mae = mean_absolute_error(all_labels, all_preds)
        val_r2 = r2_score(all_labels, all_preds)
        val_losses.append(avg_val_loss)
        val_maes.append(val_mae)
        val_r2s.append(val_r2)

        print(
            f"[{yaml_name}] Epoch {epoch + 1}/{EPOCHS} -> 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f} | 验证MAE: {val_mae:.4f} | 验证R2: {val_r2:.4f}"
        )
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), os.path.join(save_dir, "best.pt"))
            print(f"🎉 新的最佳模型已保存，验证损失: {best_val_loss:.4f}")

    print(f"\n模型 {yaml_name} 训练完成，正在保存日志...")
    with open(os.path.join(save_dir, "train_losses.txt"), "w", encoding="utf-8") as f:
        f.writelines(f"{item}\n" for item in train_losses)
    with open(os.path.join(save_dir, "val_losses.txt"), "w", encoding="utf-8") as f:
        f.writelines(f"{item}\n" for item in val_losses)
    with open(os.path.join(save_dir, "val_maes.txt"), "w", encoding="utf-8") as f:
        f.writelines(f"{item}\n" for item in val_maes)
    with open(os.path.join(save_dir, "val_r2s.txt"), "w", encoding="utf-8") as f:
        f.writelines(f"{item}\n" for item in val_r2s)
    print("所有日志已保存。")


# ===================================================================
# 5. 主执行器 (与您提供的代码保持一致)
# ===================================================================
if __name__ == "__main__":
    yaml_dir = os.path.join(PROJECT_ROOT, PROJECT_NAME, "yaml")
    yaml_files = glob.glob(os.path.join(yaml_dir, "*.yaml"))
    print(f"共检测到 {len(yaml_files)} 个yaml文件，将依次训练：")
    for yaml_path in yaml_files:
        train_one_yaml(yaml_path)
