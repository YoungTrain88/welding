import os

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(csv_path, train_ratio=0.8, random_state=42):
    """
    将CSV数据集按指定比例划分为训练集和测试集.

    Args:
        csv_path (str): 原始CSV文件路径
        train_ratio (float): 训练集比例，默认0.8 (即4:1)
        random_state (int): 随机种子，保证结果可重现
    """
    print(f"正在读取数据集: {csv_path}")

    # 读取原始数据
    df = pd.read_csv(csv_path)
    print(f"数据集总数: {len(df)} 条记录")
    print(f"目标值范围: {df['target_value'].min()} - {df['target_value'].max()}")

    # 检查数据
    print("\n数据预览:")
    print(df.head())

    # 随机划分数据集
    train_df, test_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        shuffle=True,  # 确保随机打乱
    )

    print("\n划分结果:")
    print(f"训练集: {len(train_df)} 条记录 ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"测试集: {len(test_df)} 条记录 ({len(test_df) / len(df) * 100:.1f}%)")

    # 检查目标值分布
    print("\n目标值分布:")
    print(f"训练集目标值范围: {train_df['target_value'].min()} - {train_df['target_value'].max()}")
    print(f"测试集目标值范围: {test_df['target_value'].min()} - {test_df['target_value'].max()}")
    print(f"训练集目标值均值: {train_df['target_value'].mean():.2f}")
    print(f"测试集目标值均值: {test_df['target_value'].mean():.2f}")

    # 获取保存路径
    base_dir = os.path.dirname(csv_path)
    train_path = os.path.join(base_dir, "train.csv")
    test_path = os.path.join(base_dir, "val.csv")  # 使用val.csv以匹配训练代码

    # 保存划分后的数据集
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n数据集已保存:")
    print(f"训练集: {train_path}")
    print(f"测试集: {test_path}")

    return train_df, test_df


if __name__ == "__main__":
    # 设置文件路径
    csv_file = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\其他模型-正反拼接300轮\datasets\all.csv"

    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"错误: 文件不存在 {csv_file}")
        exit(1)

    # 执行数据集划分
    try:
        train_data, test_data = split_dataset(
            csv_path=csv_file,
            train_ratio=0.8,  # 4:1 的比例
            random_state=42,  # 固定随机种子，确保结果可重现
        )
        print("\n✅ 数据集划分完成!")

    except Exception as e:
        print(f"❌ 划分过程中出现错误: {e}")
