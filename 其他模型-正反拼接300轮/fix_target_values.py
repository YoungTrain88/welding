import os

import pandas as pd


def fix_target_values(scale_up_factor=1000):
    """
    如果目标值被过度缩放，将其恢复到正常范围.

    Args:
        scale_up_factor (int): 放大因子，默认1000
    """
    # 文件路径
    project_root = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\其他模型-正反拼接300轮"
    train_csv = os.path.join(project_root, "datasets", "train.csv")
    val_csv = os.path.join(project_root, "datasets", "val.csv")

    for csv_path, name in [(train_csv, "训练集"), (val_csv, "验证集")]:
        if os.path.exists(csv_path):
            print(f"处理 {name}: {os.path.basename(csv_path)}")

            # 读取数据
            df = pd.read_csv(csv_path)
            original_max = df["target_value"].max()
            original_min = df["target_value"].min()
            original_mean = df["target_value"].mean()

            print(f"  原始范围: {original_min:.6f} - {original_max:.6f} (均值: {original_mean:.6f})")

            # 检查是否需要放大
            if original_max < 100:  # 如果最大值小于100，很可能需要放大
                # 放大目标值
                df["target_value"] = df["target_value"] * scale_up_factor

                new_max = df["target_value"].max()
                new_min = df["target_value"].min()
                new_mean = df["target_value"].mean()

                print(f"  ✅ 已放大{scale_up_factor}倍")
                print(f"  新范围: {new_min:.0f} - {new_max:.0f} (均值: {new_mean:.0f})")

                # 保存修改后的文件
                df.to_csv(csv_path, index=False)
                print("  💾 文件已更新")

            else:
                print("  ℹ️  数据范围正常，无需修改")

            print()
        else:
            print(f"❌ 文件不存在: {csv_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("目标值修复脚本")
    print("=" * 60)

    # 首先检查当前状态
    from check_data_range import check_target_values

    check_target_values()

    print("\n" + "=" * 60)
    print("开始修复...")
    print("=" * 60)

    fix_target_values(scale_up_factor=1000)
