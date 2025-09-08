import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm


# ===================================================================
# 配置参数
# ===================================================================
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    IMG_SIZE = 224

    # 模型权重路径（需要根据实际情况修改）
    MODEL_WEIGHTS_DIR = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\其他模型-正反拼接300轮"

    # 新数据集路径（需要根据实际情况修改）
    NEW_DATASET_CSV = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\新数据集\test.csv"
    NEW_DATASET_ROOT = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\新数据集"


# ===================================================================
# 模型创建函数（与训练时保持一致）
# ===================================================================
def create_regression_model(model_name: str):
    """创建回归模型，与训练时的结构保持一致."""
    print(f"--- 正在创建模型: {model_name} ---")

    if model_name == "resnet50":
        model = models.resnet50(weights=None)  # 不加载预训练权重
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)

    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 1)

    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 1)

    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, 1)

    elif "efficientnet" in model_name:
        model = timm.create_model(model_name, pretrained=False, num_classes=1)

    elif "vit" in model_name:
        model = timm.create_model(model_name, pretrained=False, num_classes=1)

    elif "resnext" in model_name:
        model = timm.create_model(model_name, pretrained=False, num_classes=1)

    elif "swin" in model_name:
        model = timm.create_model(model_name, pretrained=False, num_classes=1)

    elif "convnext" in model_name:
        model = timm.create_model(model_name, pretrained=False, num_classes=1)

    else:
        raise ValueError(f"未知的模型名称: {model_name}")

    return model


# ===================================================================
# 测试数据集类
# ===================================================================
class TestDataset:
    def __init__(self, csv_path, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.missing_files = 0

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_relative_path = self.data_frame.iloc[idx, 0]
        img_abs_path = os.path.join(self.root_dir, img_relative_path)

        target_value = float(self.data_frame.iloc[idx, 1])

        try:
            image = Image.open(img_abs_path).convert("RGB")
        except FileNotFoundError:
            print(f"⚠️  警告: 文件不存在 {img_abs_path}")
            self.missing_files += 1
            image = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE), color="black")

        value = torch.tensor([target_value], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, value


# ===================================================================
# 模型评估函数
# ===================================================================
def evaluate_model(model, test_loader, model_name):
    """评估模型性能."""
    model.eval()
    all_preds = []
    all_labels = []

    print(f"正在评估模型: {model_name}")

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"评估 {model_name}"):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # 计算评估指标
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_preds)

    # 计算平均绝对百分比误差 (MAP)
    map = np.mean(np.abs((np.array(all_labels) - np.array(all_preds)) / np.array(all_labels))) * 100

    results = {
        "model_name": model_name,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "map": map,
        "predictions": all_preds,
        "labels": all_labels,
        "pred_range": [min(all_preds), max(all_preds)],
        "label_range": [min(all_labels), max(all_labels)],
    }

    return results


# ===================================================================
# 可视化函数
# ===================================================================
def plot_predictions(results, save_path=None):
    """绘制预测值 vs 真实值的散点图."""
    plt.figure(figsize=(10, 8))

    labels = results["labels"]
    predictions = results["predictions"]
    model_name = results["model_name"]

    # 散点图
    plt.scatter(labels, predictions, alpha=0.6, s=50)

    # 完美预测线
    min_val = min(min(labels), min(predictions))
    max_val = max(max(labels), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="完美预测线")

    # 添加统计信息
    plt.text(
        0.05,
        0.95,
        f"R² = {results['r2']:.4f}\n"
        f"MAE = {results['mae']:.4f}\n"
        f"RMSE = {results['rmse']:.4f}\n"
        f"MAP = {results['mape']:.2f}%",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.xlabel("真实值", fontsize=12)
    plt.ylabel("预测值", fontsize=12)
    plt.title(f"模型 {model_name} 预测结果", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_error_distribution(results, save_path=None):
    """绘制误差分布图."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    labels = np.array(results["labels"])
    predictions = np.array(results["predictions"])
    errors = predictions - labels

    # 误差直方图
    ax1.hist(errors, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    ax1.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax1.set_xlabel("预测误差 (预测值 - 真实值)")
    ax1.set_ylabel("频次")
    ax1.set_title(f"误差分布 - {results['model_name']}")
    ax1.grid(True, alpha=0.3)

    # 残差 vs 预测值
    ax2.scatter(predictions, errors, alpha=0.6)
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("预测值")
    ax2.set_ylabel("残差 (预测值 - 真实值)")
    ax2.set_title(f"残差图 - {results['model_name']}")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ===================================================================
# 批量评估函数
# ===================================================================
def evaluate_all_models(test_dataset, available_models, output_dir="evaluation_results"):
    """批量评估所有可用模型."""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 数据变换（与训练时保持一致）
    transform = transforms.Compose(
        [
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 创建测试数据集
    test_data = TestDataset(csv_path=test_dataset, root_dir=os.path.dirname(test_dataset), transform=transform)

    test_loader = DataLoader(test_data, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("测试数据集信息:")
    print(f"- 样本数量: {len(test_data)}")
    print(f"- 缺失文件: {test_data.missing_files}")

    all_results = []

    # 评估每个模型
    for model_name in available_models:
        try:
            print(f"\n{'=' * 50}")
            print(f"评估模型: {model_name}")
            print(f"{'=' * 50}")

            # 构造权重文件路径
            weight_path = os.path.join(Config.MODEL_WEIGHTS_DIR, f"runs_{model_name}", "best.pt")

            if not os.path.exists(weight_path):
                print(f"❌ 权重文件不存在: {weight_path}")
                continue

            # 创建并加载模型
            model = create_regression_model(model_name)
            model.load_state_dict(torch.load(weight_path, map_location=Config.DEVICE))
            model.to(Config.DEVICE)

            # 评估模型
            results = evaluate_model(model, test_loader, model_name)
            all_results.append(results)

            # 打印结果
            print("结果:")
            print(f"  MAE:  {results['mae']:.4f}")
            print(f"  RMSE: {results['rmse']:.4f}")
            print(f"  R²:   {results['r2']:.4f}")
            print(f"  MAP: {results['mape']:.2f}%")
            print(f"  预测范围: [{results['pred_range'][0]:.3f}, {results['pred_range'][1]:.3f}]")
            print(f"  真实范围: [{results['label_range'][0]:.3f}, {results['label_range'][1]:.3f}]")

            # 保存可视化结果
            plot_predictions(results, os.path.join(output_dir, f"{model_name}_predictions.png"))
            plot_error_distribution(results, os.path.join(output_dir, f"{model_name}_errors.png"))

            # 清理显存
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ 评估模型 {model_name} 时出错: {e}")
            continue

    # 生成综合报告
    if all_results:
        generate_comparison_report(all_results, output_dir)

    return all_results


def generate_comparison_report(all_results, output_dir):
    """生成模型比较报告."""
    # 创建比较表格
    comparison_data = []
    for result in all_results:
        comparison_data.append(
            {
                "模型名称": result["model_name"],
                "MAE": f"{result['mae']:.4f}",
                "RMSE": f"{result['rmse']:.4f}",
                "R²": f"{result['r2']:.4f}",
                "MAP(%)": f"{result['mape']:.2f}",
            }
        )

    df_comparison = pd.DataFrame(comparison_data)

    # 保存到CSV
    df_comparison.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    # 打印比较表格
    print(f"\n{'=' * 80}")
    print("模型性能比较")
    print(f"{'=' * 80}")
    print(df_comparison.to_string(index=False))

    # 绘制性能比较图
    plt.figure(figsize=(15, 10))

    metrics = ["MAE", "RMSE", "R²", "MAP(%)"]
    model_names = [r["model_name"] for r in all_results]

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)

        if metric == "MAP(%)":
            values = [r["map"] for r in all_results]
        elif metric == "MAE":
            values = [r["mae"] for r in all_results]
        elif metric == "RMSE":
            values = [r["rmse"] for r in all_results]
        elif metric == "R²":
            values = [r["r2"] for r in all_results]

        bars = plt.bar(model_names, values)
        plt.title(f"{metric} 比较")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(metric)

        # 添加数值标签
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n✅ 评估完成！结果保存在: {output_dir}")


# ===================================================================
# 主函数
# ===================================================================
if __name__ == "__main__":
    # 可用的模型列表（需要确保对应的权重文件存在）
    available_models = [
        "resnet50",
        "densenet121",
        "mobilenet_v3_large",
        "vgg16",
        "efficientnet_b0",
        "resnext50_32x4d",
        "convnext_tiny",
        "vit_base_patch16_224",
        "swin_tiny_patch4_window7_224",
    ]

    print("=" * 80)
    print("模型性能评估脚本")
    print("=" * 80)

    # 检查配置
    print(f"设备: {Config.DEVICE}")
    print(f"测试数据集: {Config.NEW_DATASET_CSV}")
    print(f"模型权重目录: {Config.MODEL_WEIGHTS_DIR}")

    # 检查测试数据集是否存在
    if not os.path.exists(Config.NEW_DATASET_CSV):
        print(f"❌ 测试数据集不存在: {Config.NEW_DATASET_CSV}")
        print("请修改 Config.NEW_DATASET_CSV 为正确的路径")
        exit(1)

    # 开始评估
    results = evaluate_all_models(
        test_dataset=Config.NEW_DATASET_CSV, available_models=available_models, output_dir="evaluation_results"
    )

    if not results:
        print("❌ 没有成功评估任何模型")
    else:
        print(f"✅ 成功评估了 {len(results)} 个模型")
