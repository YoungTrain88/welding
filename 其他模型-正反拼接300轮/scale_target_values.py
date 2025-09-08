import os

import pandas as pd


def scale_target_values(csv_file_path, scale_factor=0.001, backup=True):
    """
    将CSV文件中的target_value列缩小指定倍数.

    Args:
        csv_file_path (str): CSV文件路径
        scale_factor (float): 缩放因子，默认0.001（缩小1000倍，即缩小三位小数点）
        backup (bool): 是否创建备份文件
    """
    print(f"正在处理文件: {csv_file_path}")

    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件不存在 {csv_file_path}")
        return

    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    print("原始数据统计:")
    print(f"- 总记录数: {len(df)}")
    print(f"- target_value 范围: {df['target_value'].min()} - {df['target_value'].max()}")
    print(f"- target_value 均值: {df['target_value'].mean():.2f}")

    # 创建备份（如果需要）
    if backup:
        backup_path = csv_file_path.replace(".csv", "_backup.csv")
        df.to_csv(backup_path, index=False)
        print(f"备份文件已创建: {backup_path}")

    # 缩放target_value列
    df["target_value"] = df["target_value"] * scale_factor

    print(f"\n缩放后数据统计 (缩放因子: {scale_factor}):")
    print(f"- target_value 范围: {df['target_value'].min():.3f} - {df['target_value'].max():.3f}")
    print(f"- target_value 均值: {df['target_value'].mean():.3f}")

    # 保存修改后的文件
    df.to_csv(csv_file_path, index=False)
    print(f"✅ 文件已更新: {csv_file_path}")

    return df


def batch_scale_csv_files(directory_path, scale_factor=0.001):
    """
    批量处理目录下的所有CSV文件.

    Args:
        directory_path (str): 目录路径
        scale_factor (float): 缩放因子
    """
    csv_files = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".csv") and not file_name.endswith("_backup.csv"):
            csv_files.append(os.path.join(directory_path, file_name))

    if not csv_files:
        print(f"在目录 {directory_path} 中未找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件:")
    for csv_file in csv_files:
        print(f"- {os.path.basename(csv_file)}")

    for csv_file in csv_files:
        print(f"\n{'=' * 60}")
        scale_target_values(csv_file, scale_factor)


if __name__ == "__main__":
    # 设置数据集目录路径
    dataset_dir = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\其他模型-正反拼接300轮\datasets"

    # 批量处理所有CSV文件
    print("开始批量处理CSV文件...")
    batch_scale_csv_files(dataset_dir, scale_factor=0.001)

    print("\n" + "=" * 60)
    print("✅ 所有文件处理完成!")
    print("\n处理结果:")
    print("- 所有target_value值已缩小1000倍")
    print("- 原始文件已备份为 *_backup.csv")
    print("- 现在目标值范围大约在 1.0 - 8.0 之间，更适合模型训练")
