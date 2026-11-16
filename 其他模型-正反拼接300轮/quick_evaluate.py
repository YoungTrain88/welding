import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import transforms

# ===================================================================
# 快速单模型评估脚本
# ===================================================================


def quick_evaluate_single_model():
    """快速评估单个模型."""
    # ===== 配置 =====
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 224
    BATCH_SIZE = 16

    # 🔧 需要修改的路径
    MODEL_NAME = "resnet50"  # 要评估的模型名称
    WEIGHT_PATH = (
        r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\其他模型-正反拼接300轮\runs_resnet50\best.pt"
    )
    TEST_CSV = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\其他模型-正面300轮\datasets\val.csv"
    TEST_ROOT = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\其他模型-正面300轮"

    print("快速模型评估")
    print(f"模型: {MODEL_NAME}")
    print(f"权重: {WEIGHT_PATH}")
    print(f"测试数据: {TEST_CSV}")
    print(f"设备: {DEVICE}")
    print("-" * 50)

    # ===== 检查文件 =====
    if not os.path.exists(WEIGHT_PATH):
        print(f"❌ 权重文件不存在: {WEIGHT_PATH}")
        return

    if not os.path.exists(TEST_CSV):
        print(f"❌ 测试数据不存在: {TEST_CSV}")
        return

    # ===== 创建模型 =====
    print("正在创建模型...")
    if MODEL_NAME == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif MODEL_NAME == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    elif MODEL_NAME == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    elif "efficientnet" in MODEL_NAME:
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1)
    else:
        print(f"❌ 不支持的模型: {MODEL_NAME}")
        return

    # ===== 加载权重 =====
    print("正在加载权重...")
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # ===== 创建数据集 =====
    class QuickTestDataset(Dataset):
        def __init__(self, csv_path, root_dir, transform=None):
            self.df = pd.read_csv(csv_path)
            self.root_dir = root_dir
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            img_path = os.path.join(self.root_dir, self.df.iloc[idx, 0])
            target = float(self.df.iloc[idx, 1])

            try:
                image = Image.open(img_path).convert("RGB")
            except:
                image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "black")

            if self.transform:
                image = self.transform(image)

            return image, torch.tensor([target], dtype=torch.float32)

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = QuickTestDataset(TEST_CSV, TEST_ROOT, transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"测试样本数: {len(test_dataset)}")

    # ===== 评估模型 =====
    print("正在评估...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # ===== 计算指标 =====
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_preds)
    map = np.mean(np.abs((all_labels - all_preds) / all_labels)) * 100

    # ===== 打印结果 =====
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"平均绝对误差 (MAE):     {mae:.4f}")
    print(f"均方根误差 (RMSE):      {rmse:.4f}")
    print(f"决定系数 (R²):         {r2:.4f}")
    print(f"平均绝对百分比误差:     {map:.2f}%")
    print(f"预测值范围:           [{all_preds.min():.3f}, {all_preds.max():.3f}]")
    print(f"真实值范围:           [{all_labels.min():.3f}, {all_labels.max():.3f}]")

    # ===== 可视化 =====
    plt.figure(figsize=(12, 5))

    # 散点图
    plt.subplot(1, 2, 1)
    plt.scatter(all_labels, all_preds, alpha=0.6, s=20)
    min_val = min(all_labels.min(), all_preds.min())
    max_val = max(all_labels.max(), all_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title(f"{MODEL_NAME} - 预测 vs 真实值")
    plt.grid(True, alpha=0.3)

    # 误差分布
    plt.subplot(1, 2, 2)
    errors = all_preds - all_labels
    plt.hist(errors, bins=20, alpha=0.7, color="lightblue", edgecolor="black")
    plt.axvline(0, color="red", linestyle="--", linewidth=2)
    plt.xlabel("预测误差")
    plt.ylabel("频次")
    plt.title(f"{MODEL_NAME} - 误差分布")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{MODEL_NAME}_evaluation.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ===== 保存详细结果 =====
    results_df = pd.DataFrame(
        {"true_values": all_labels, "predictions": all_preds, "errors": errors, "abs_errors": np.abs(errors)}
    )
    results_df.to_csv(f"{MODEL_NAME}_detailed_results.csv", index=False)

    print("\n✅ 评估完成！")
    print(f"📊 图表已保存: {MODEL_NAME}_evaluation.png")
    print(f"📄 详细结果已保存: {MODEL_NAME}_detailed_results.csv")


if __name__ == "__main__":
    quick_evaluate_single_model()
