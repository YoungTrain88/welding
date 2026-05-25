import glob
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from torchvision.transforms import transforms
from tqdm import tqdm

# 导入自定义-RegressionModel
from yolo8_12_fb_all_net_MultiModal.custom_modules.custom_tasks import RegressionModel

# 配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# MODEL_YAML_PATH = r'C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\my_yolo_r_p1_c_s_att_conv\yoloV11n-r-att-conv.yaml'
# PRETRAINED_WEIGHTS_PATH = 'yolo11n-cls.pt'  # 你的预训练权重
PROJECT_NAME = "yolo8_12_f_all_net_MultiModal"
EPOCHS = 100  # 调整训练轮数看R2能否有所提高
TRAIN_CSV_PATH = os.path.join(PROJECT_ROOT, PROJECT_NAME, "datasets", "train.csv")
VAL_CSV_PATH = os.path.join(PROJECT_ROOT, PROJECT_NAME, "datasets", "val.csv")
# SAVE_DIR = os.path.join('my_yolo_r_p1_c_s_att_conv', 'runs-yolo11n-AFAR')  # 可自定义

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
LEARNING_RATE = 1e-5  # 把学习率再调低到0.00001，看训练曲线是否还会一直震荡

IMG_SIZE = 224

# os.makedirs(SAVE_DIR, exist_ok=True)

# class RegressionDataset(Dataset):
#     def __init__(self, csv_path, transform=None):
#         self.data_frame = pd.read_csv(csv_path)
#         self.transform = transform

#     def __len__(self):
#         return len(self.data_frame)

#     def __getitem__(self, idx):
#         img_relative_path = self.data_frame.iloc[idx, 0]
#         img_abs_path = os.path.join(PROJECT_ROOT, PROJECT_NAME, img_relative_path)
#         image = Image.open(img_abs_path).convert("RGB")
#         value = torch.tensor([self.data_frame.iloc[idx, 1]], dtype=torch.float32)
#         if self.transform:
#             image = self.transform(image)
#         return image, value


class RegressionDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data_frame = pd.read_csv(csv_path)
        self.transform = transform
        self.tabular_data = self.data_frame.iloc[:, 2:].values.astype("float32")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_relative_path = self.data_frame.iloc[idx, 0]
        img_abs_path = os.path.join(PROJECT_ROOT, PROJECT_NAME, img_relative_path)
        image = Image.open(img_abs_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        tabular_features = torch.tensor(self.tabular_data[idx], dtype=torch.float32)
        target_value = torch.tensor([self.data_frame.iloc[idx, 1]], dtype=torch.float32)

        return image, tabular_features, target_value


def train_one_yaml(yaml_path):
    print(f"\n========== 开始训练: {yaml_path} ==========")
    # 保存路径改为yolo8-12-正面/runs-xxx
    yaml_name = os.path.splitext(os.path.basename(yaml_path))[0]
    save_dir = os.path.join(PROJECT_ROOT, PROJECT_NAME, f"runs-{yaml_name}")
    os.makedirs(save_dir, exist_ok=True)

    # 其余配置和数据加载不变
    # transform = transforms.Compose([
    #     transforms.Resize((IMG_SIZE, IMG_SIZE)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),  # <--- 新增：随机水平翻转
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # <--- 新增：随机颜色抖动
            transforms.RandomRotation(10),  # <--- 新增：随机旋转 +/-10度
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [  # 验证集的变换保持不变
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = RegressionDataset(csv_path=TRAIN_CSV_PATH, transform=train_transform)
    val_dataset = RegressionDataset(csv_path=VAL_CSV_PATH, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 确定你的表格数据有多少个特征
    # 例如，从CSV文件中动态获取
    temp_df = pd.read_csv(TRAIN_CSV_PATH)
    NUM_TABULAR_FEATURES = len(temp_df.columns) - 2  # 减去 path 和 target
    print(f"检测到 {NUM_TABULAR_FEATURES} 个表格特征。")

    # 初始化新的多模态模型
    net = MultiModalModel(yaml_path=yaml_path, num_tabular_features=NUM_TABULAR_FEATURES).to(DEVICE)
    # 获取表格特征数量
    temp_df = pd.read_csv(TRAIN_CSV_PATH)
    NUM_TABULAR_FEATURES = len(temp_df.columns) - 2
    print(f"检测到 {NUM_TABULAR_FEATURES} 个表格特征。")

    # 初始化新的多模态模型
    net = MultiModalModel(yaml_path=yaml_path, num_tabular_features=NUM_TABULAR_FEATURES).to(DEVICE)

    # ==================== 在这里添加模型结构打印代码 ====================
    print("\n" + "=" * 50)
    print(f"             模型结构: {os.path.basename(yaml_path)}")
    print("=" * 50)

    # 定义模型的输入尺寸，注意我们有两个输入，所以提供一个列表
    # (batch_size, channels, height, width) for image
    # (batch_size, num_features) for tabular data
    # IMG_SIZE 是您在脚本配置中定义的图像尺寸
    input_sizes = [(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE), (BATCH_SIZE, NUM_TABULAR_FEATURES)]

    # 使用 torchinfo 打印详细摘要
    # col_names 指定了要显示的列
    summary(
        net, input_size=input_sizes, col_names=["input_size", "output_size", "num_params", "mult_adds"], depth=3
    )  # depth 控制显示子模块的深度

    print("=" * 50 + "\n")
    # ========================== 打印代码结束 ==========================

    criterion = nn.MSELoss()
    # 修改这里，加入 weight_decay
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)  # <--- 增加 weight_decay
    # ... (scheduler 和其他变量定义不变) ...
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float("inf")
    train_losses, val_losses, val_maes, val_r2s = [], [], [], []

    for epoch in range(EPOCHS):
        net.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"[{yaml_name}] Epoch {epoch + 1}/{EPOCHS} [训练]")
        # 修改数据加载循环
        for images, tabular, labels in train_pbar:
            images, tabular, labels = images.to(DEVICE), tabular.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            # 将图像和表格数据同时传入模型
            outputs = net(images, tabular)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # ... (更新 loss 和进度条的逻辑不变) ...
            running_loss += loss.item()
            train_pbar.set_postfix({"loss": loss.item()})
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        net.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"[{yaml_name}] Epoch {epoch + 1}/{EPOCHS} [验证]")
            # 修改验证循环
            for images, tabular, labels in val_pbar:
                images, tabular, labels = images.to(DEVICE), tabular.to(DEVICE), labels.to(DEVICE)

                # 将图像和表格数据同时传入模型
                outputs = net(images, tabular)

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
        for mae in val_maes:
            f.write(f"{mae}\n")
    with open(os.path.join(save_dir, "val_r2s.txt"), "w", encoding="utf-8") as f:
        for r2 in val_r2s:
            f.write(f"{r2}\n")


class MultiModalModel(nn.Module):
    def __init__(self, yaml_path, num_tabular_features, hidden_dim=64):
        """
        Args:
            yaml_path (str): 图像模型的yaml配置文件路径.
            num_tabular_features (int): 输入的表格/数值特征的数量.
            hidden_dim (int): MLP和融合头的隐藏层维度.
        """
        super().__init__()

        # 1. 图像分支: 加载我们修改过的RegressionModel
        # 我们将以“特征提取模式”来使用它
        self.image_branch = RegressionModel(yaml_path, ch=3, verbose=False)  # verbose=False避免重复打印模型结构

        # 从RegressionHead中我们知道图像特征维度是1280
        img_feature_dim = 1280

        # 2. 表格数据分支 (一个简单的MLP)
        self.tabular_branch = nn.Sequential(
            nn.Linear(num_tabular_features, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),  # <--- 从 0.2 增加到 0.5
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # 3. 融合后的回归头
        self.fusion_head = nn.Sequential(
            nn.Linear(img_feature_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),  # <--- 从 0.2 增加到 0.5
            nn.Linear(hidden_dim, 1),  # 最后输出一个回归值
        )

    def forward(self, image, tabular_data):
        # 1. 从图像分支获取图像特征向量 (1280维)
        # 调用时传入 return_features=True
        image_features = self.image_branch(image, return_features=True)

        # 2. 从表格数据分支获取特征
        tabular_features = self.tabular_branch(tabular_data)

        # 3. 融合特征 (拼接)
        combined_features = torch.cat([image_features, tabular_features], dim=1)

        # 4. 通过回归头得到最终结果
        output = self.fusion_head(combined_features)
        return output


if __name__ == "__main__":
    yaml_dir = os.path.join(PROJECT_ROOT, PROJECT_NAME, "yaml")
    yaml_files = glob.glob(os.path.join(yaml_dir, "*.yaml"))
    print(f"共检测到 {len(yaml_files)} 个yaml文件，将依次训练：")
    for yaml_path in yaml_files:
        train_one_yaml(yaml_path)
