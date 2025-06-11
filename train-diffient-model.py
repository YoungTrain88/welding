# run_experiments.py

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
from torchvision.transforms import transforms
from tqdm import tqdm

# ===================================================================
# 1. 全局配置 (基本不变)
# ===================================================================

PROJECT_ROOT = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\my_yolo_regression_project1-cat-shuffed"
TRAIN_CSV_PATH = os.path.join(PROJECT_ROOT, "datasets", "train.csv")
VAL_CSV_PATH = os.path.join(PROJECT_ROOT, "datasets", "val.csv")

# 训练超参数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50  # 建议先用较小的epoch（如50）快速迭代，找到好模型后再增加
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
IMG_SIZE = 224


# ===================================================================
# 2. 模型工厂：根据名称创建不同的回归模型
# ===================================================================
def create_regression_model(model_name: str):
    """
    根据模型名称，创建并返回一个修改好用于回归的预训练模型。.

    Args:
        model_name (str): 模型的名称, e.g., 'resnet50', 'efficientnet_b0'.

    Returns:
        torch.nn.Module: 一个准备好进行回归训练的模型。
    """
    print(f"--- 正在创建模型: {model_name} ---")

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
    # timm的create_model接口非常统一，可以直接用num_classes=1来创建回归头
    elif "efficientnet" in model_name:
        model = timm.create_model(model_name, pretrained=True, num_classes=1)

    elif "vit" in model_name:  # Vision Transformer
        model = timm.create_model(model_name, pretrained=True, num_classes=1)

    elif "resnext" in model_name:
        model = timm.create_model(model_name, pretrained=True, num_classes=1)

    elif "swin" in model_name:  # Swin Transformer
        model = timm.create_model(model_name, pretrained=True, num_classes=1)

    elif "convnext" in model_name:  # ConvNeXt
        model = timm.create_model(model_name, pretrained=True, num_classes=1)

    else:
        raise ValueError(f"未知的模型名称: {model_name}")

    return model


# ===================================================================
# 3. 数据集类 (保持不变)
# ===================================================================
class RegressionDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data_frame = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_relative_path = self.data_frame.iloc[idx, 0]
        img_abs_path = os.path.join(PROJECT_ROOT, img_relative_path)
        try:
            image = Image.open(img_abs_path).convert("RGB")
        except FileNotFoundError:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), torch.tensor([0.0])
        value = torch.tensor([self.data_frame.iloc[idx, 1]], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, value


# ===================================================================
# 4. 通用训练函数 (从main函数重构而来)
# ===================================================================
def run_training_session(model, model_name, train_loader, val_loader, save_dir):
    """对给定的模型执行一个完整的训练和验证流程。."""
    print(f"模型已移至设备: {DEVICE}")
    model.to(DEVICE)

    # --- 损失函数和优化器 ---
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # --- 训练循环 ---
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    val_maes = []
    val_r2s = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"模型: {model_name} | Epoch {epoch + 1}/{EPOCHS} [训练]")
        for images, labels in train_pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- 验证阶段 ---
        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_mae = mean_absolute_error(all_labels, all_preds)
        val_r2 = r2_score(all_labels, all_preds)
        val_maes.append(val_mae)
        val_r2s.append(val_r2)

        print(
            f"结果 -> 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f} | 验证MAE: {val_mae:.4f} | 验证R2: {val_r2:.4f}"
        )
        scheduler.step()

        # --- 保存模型 ---
        torch.save(model.state_dict(), os.path.join(save_dir, "last.pt"))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
            print(f"🎉 模型 {model_name} 发现新的最佳权重, 验证损失: {best_val_loss:.4f}")

    print(f"\n模型 {model_name} 训练完成！")
    print(f"最佳模型权重已保存到: {os.path.join(save_dir, 'best.pt')}")

    # 保存每一轮的损失和指标
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


# ===================================================================
# 5. 主执行器
# ===================================================================
if __name__ == "__main__":
    # 🚀 在这里定义您想进行对比实验的所有模型！
    # 您可以注释掉不想跑的模型，或者添加timm库支持的其他模型名。
    models_to_train = [
        # --- 经典模型 (来自 torchvision) ---
        "resnet50",  # 经典基准
        "densenet121",  # 密集连接，参数高效
        "mobilenet_v3_large",  # 优秀的轻量级模型
        "vgg16",  # 结构简单，深度学习的早期里程碑
        # --- 现代SOTA模型 (来自 timm) ---
        "efficientnet_b0",  # 性能与效率的完美平衡
        "resnext50_32x4d",  # ResNet的强大改进版
        "convnext_tiny",  # 现代化的纯卷积模型
        # --- Transformer架构 ---
        "vit_base_patch16_224",  # Vision Transformer 基础版
        "swin_tiny_patch4_window7_224",  # Swin Transformer, 对ViT的改进
        # 您还可以添加更多timm支持的模型...
        # 'efficientnet_b2',
        # 'regnetx_002',
    ]

    # --- 数据加载 (只需要执行一次) ---
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
    print("数据加载器已创建，准备开始实验...")

    # --- 循环执行所有实验 ---
    for model_name in models_to_train:
        print(f"\n{'=' * 25} 正在开始模型: {model_name.upper()} 的实验 {'=' * 25}")

        # 为每个模型创建独立的保存目录
        current_save_dir = os.path.join(PROJECT_ROOT, f"runs_{model_name}")
        os.makedirs(current_save_dir, exist_ok=True)

        # 创建模型
        model = create_regression_model(model_name)

        # 执行训练
        run_training_session(model, model_name, train_loader, val_loader, current_save_dir)

        # (可选) 清理显存
        del model
        torch.cuda.empty_cache()

    print("\n所有实验已完成！")
