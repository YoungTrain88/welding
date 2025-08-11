# test_dataset.py

import os

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

# ==================== 主要修改部分 ====================
# 获取此脚本文件所在的目录，也就是你的项目根目录
# os.path.abspath(__file__) -> 获取此文件的绝对路径
# os.path.dirname(...)      -> 获取该路径所在的目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# ======================================================


class RegressionDataset(Dataset):
    def __init__(self, csv_path, transform=None):  # 直接接收完整的 CSV 路径
        """
        Args:
            csv_path (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # ==================== 主要修改部分 ====================
        # 使用项目根目录构建图片的绝对路径
        img_relative_path = self.data_frame.iloc[idx, 0]
        img_abs_path = os.path.join(PROJECT_ROOT, img_relative_path)
        # ======================================================

        image = Image.open(img_abs_path).convert("RGB")

        value = torch.tensor([self.data_frame.iloc[idx, 1]], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, value


# 使用示例
if __name__ == "__main__":
    # 定义图像预处理
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # ==================== 主要修改部分 ====================
    # 构建 train.csv 文件的绝对路径
    train_csv_abs_path = os.path.join(PROJECT_ROOT, "datasets", "train.csv")
    # ======================================================

    # 创建数据集实例
    train_dataset = RegressionDataset(csv_path=train_csv_abs_path, transform=transform)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    # 测试一下数据加载器
    print("成功加载数据集！正在测试 Dataloader...")
    images, values = next(iter(train_loader))
    print("Batch of images shape:", images.shape)
    images, values = next(iter(train_loader))
    print("Batch of values shape:", values.shape)
    print("Values:", values.squeeze())
