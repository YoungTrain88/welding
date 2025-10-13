# run_experiments_multimodal_late_fusion.py

import os

import pandas as pd
import timm  # 导入timm库
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary  # 导入torchinfo库用于打印模型结构
from torchvision.transforms import transforms
from tqdm import tqdm

# ===================================================================
# 1. 全局配置
# ===================================================================

# --- 请根据您的项目路径进行修改 ---
PROJECT_ROOT = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\other_net_fb_MultiModal"
# 建议使用我们之前为解决数据分布问题而重新划分好的数据
TRAIN_CSV_PATH = os.path.join(PROJECT_ROOT, "datasets", "train.csv")
VAL_CSV_PATH = os.path.join(PROJECT_ROOT, "datasets", "val.csv")
# ---------------------------------

# --- 训练超参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 900  # 先用100轮观察趋势
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # 加入权重衰减以对抗过拟合
IMG_SIZE = 224


# ===================================================================
# 2. 多模态数据集类 (已升级)
# ===================================================================
class MultiModalDataset(Dataset):
    """一个可以同时处理图像和多列表格数据的Dataset类。."""

    def __init__(self, csv_path, project_root, transform=None):
        self.data_frame = pd.read_csv(csv_path).dropna()  # 加载并丢弃空行
        self.transform = transform
        self.project_root = project_root

        # 自动获取目标值和所有表格特征
        self.target_values = self.data_frame.iloc[:, 1].values.astype("float32")
        self.tabular_data = self.data_frame.iloc[:, 2:].values.astype("float32")
        self.image_paths = self.data_frame.iloc[:, 0].values

        self.num_tabular_features = self.tabular_data.shape[1]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # --- 稳健的路径处理 ---
        img_relative_path = self.image_paths[idx].replace("\\", "/")
        img_abs_path = os.path.join(self.project_root, img_relative_path)

        try:
            image = Image.open(img_abs_path).convert("RGB")
        except FileNotFoundError:
            print(f"警告: 找不到文件 {img_abs_path}，将返回一个零张量。")
            image = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            tabular_features = torch.zeros(self.num_tabular_features)
            target_value = torch.tensor([0.0])
            return image, tabular_features, target_value

        tabular_features = torch.tensor(self.tabular_data[idx], dtype=torch.float32)
        target_value = torch.tensor([self.target_values[idx]], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, tabular_features, target_value


# ===================================================================
# 3. 新的多模态模型定义 (后期融合策略)
# ===================================================================
class LateFusionMultiModalNet(nn.Module):
    """一个通用的“后期融合”多模态网络。 它接收一个完整的图像回归模型作为图像分支。."""

    def __init__(self, image_regression_model, num_tabular_features, hidden_dim=64, dropout_rate=0.5):
        super().__init__()
        # 图像分支：直接使用您传入的、完整的、端到端的图像回归模型
        self.image_branch = image_regression_model

        # 表格数据分支 (MLP)
        self.tabular_branch = nn.Sequential(
            nn.Linear(num_tabular_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # 融合与回归头
        # 输入维度 = 图像分支的输出(1) + 表格分支的输出(hidden_dim // 2)
        fusion_input_dim = 1 + (hidden_dim // 2)
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 最终输出一个回归值
        )

    def forward(self, image, tabular_data):
        # 1. 图像分支直接产生一个回归预测值
        image_output = self.image_branch(image)  # Shape: [batch, 1]

        # 2. 提取表格特征
        tabular_features = self.tabular_branch(tabular_data)  # Shape: [batch, 32]

        # 3. 拼接融合：将图像的预测值和表格特征拼接
        combined_features = torch.cat([image_output, tabular_features], dim=1)  # Shape: [batch, 33]

        # 4. 最终决策
        final_output = self.fusion_head(combined_features)
        return final_output


# ===================================================================
# 4. 原始的图像回归模型工厂 (完全遵循您的代码)
# ===================================================================
def create_regression_model(model_name: str):
    """根据模型名称，创建并返回一个修改好用于回归的预训练模型。."""
    print(f"--- 正在创建图像回归模型: {model_name} ---")

    # 方案A：使用 torchvision 加载经典模型
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)

    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 1)

    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 1)

    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, 1)

    # 方案B：使用 timm 加载更多、更现代的模型 (更推荐)
    elif any(name in model_name for name in ["efficientnet", "vit", "resnext", "swin", "convnext"]):
        model = timm.create_model(model_name, pretrained=True, num_classes=1)

    else:
        raise ValueError(f"未知的模型名称: {model_name}")

    return model


# ===================================================================
# 5. 通用训练函数 (已改造, 日志记录更稳健)
# ===================================================================
def run_training_session(model, model_name, train_loader, val_loader, save_dir):
    """对给定的多模态模型执行一个完整的训练和验证流程。."""
    model.to(DEVICE)
    print(f"模型 '{model_name}' 已移至设备: {DEVICE}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.5, verbose=True)

    best_val_loss = float("inf")

    # --- 关键修改：在训练开始前，先清空或创建日志文件 ---
    log_files = {
        "train_losses": os.path.join(save_dir, "train_losses.txt"),
        "val_losses": os.path.join(save_dir, "val_losses.txt"),
        "val_maes": os.path.join(save_dir, "val_maes.txt"),
        "val_r2s": os.path.join(save_dir, "val_r2s.txt"),
    }
    for file_path in log_files.values():
        with open(file_path, "w") as f:
            pass  # 创建或清空文件

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"模型: {model_name} | Epoch {epoch + 1}/{EPOCHS} [训练]")

        for images, tabular, labels in train_pbar:
            images, tabular, labels = images.to(DEVICE), tabular.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images, tabular)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for images, tabular, labels in val_loader:
                images, tabular, labels = images.to(DEVICE), tabular.to(DEVICE), labels.to(DEVICE)
                outputs = model(images, tabular)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(val_loader)
        val_mae = mean_absolute_error(all_labels, all_preds)
        val_r2 = r2_score(all_labels, all_preds)

        print(
            f"结果 -> 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f} | 验证MAE: {val_mae:.4f} | 验证R2: {val_r2:.4f}"
        )

        # --- 关键修改：在每个epoch结束后，立即将结果追加写入文件 ---
        with open(log_files["train_losses"], "a") as f:
            f.write(f"{avg_train_loss}\n")
        with open(log_files["val_losses"], "a") as f:
            f.write(f"{avg_val_loss}\n")
        with open(log_files["val_maes"], "a") as f:
            f.write(f"{val_mae}\n")
        with open(log_files["val_r2s"], "a") as f:
            f.write(f"{val_r2}\n")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
            print(f"🎉 模型 {model_name} 发现新的最佳权重, 验证损失: {best_val_loss:.4f}")

    print(f"\n模型 {model_name} 训练完成！")


# ===================================================================
# 6. 主执行器 (已改造)
# ===================================================================
if __name__ == "__main__":
    # 🚀 在这里定义您想进行对比实验的所有模型！
    models_to_train = [
        "resnet50",
        # 'efficientnet_b0',
        "convnext_tiny",
        "vit_base_patch16_224",
        # 'swin_tiny_patch4_window7_224',
    ]

    # --- 数据加载与准备 ---
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

    train_dataset = MultiModalDataset(csv_path=TRAIN_CSV_PATH, project_root=PROJECT_ROOT, transform=train_transform)
    val_dataset = MultiModalDataset(csv_path=VAL_CSV_PATH, project_root=PROJECT_ROOT, transform=val_transform)

    num_tabular_features = train_dataset.num_tabular_features

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"多模态数据加载器已创建，共检测到 {num_tabular_features} 个表格特征。准备开始实验...")

    # --- 循环执行所有实验 ---
    for model_name in models_to_train:
        print(f"\n{'=' * 25} 正在开始模型: {model_name.upper()} 的实验 {'=' * 25}")

        current_save_dir = os.path.join(PROJECT_ROOT, f"runs_multimodal_{model_name}")
        os.makedirs(current_save_dir, exist_ok=True)

        # 1. 严格按照您的要求，创建原始的、端到端的图像回归模型
        image_regression_model = create_regression_model(model_name)

        # 2. 将这个图像回归模型作为“图像分支”，组装成最终的“后期融合”多模态模型
        late_fusion_model = LateFusionMultiModalNet(
            image_regression_model=image_regression_model, num_tabular_features=num_tabular_features
        )

        # (可选) 打印模型结构
        print("\n--- 模型结构摘要 ---")
        summary(
            late_fusion_model,
            input_size=[(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE), (BATCH_SIZE, num_tabular_features)],
            depth=3,
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
        )
        print("---------------------\n")

        # 3. 执行训练
        run_training_session(late_fusion_model, model_name, train_loader, val_loader, current_save_dir)

        del image_regression_model, late_fusion_model
        torch.cuda.empty_cache()

    print("\n所有实验已完成！")
