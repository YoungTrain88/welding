import pandas as pd
import os

def check_target_values():
    """检查当前CSV文件中的目标值范围"""
    
    # 文件路径
    project_root = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\其他模型-正反拼接300轮"
    train_csv = os.path.join(project_root, 'datasets', 'train.csv')
    val_csv = os.path.join(project_root, 'datasets', 'val.csv')
    
    print("=" * 60)
    print("当前数据集目标值分析")
    print("=" * 60)
    
    for csv_path, name in [(train_csv, "训练集"), (val_csv, "验证集")]:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            target_values = df['target_value'].values
            
            print(f"\n{name} ({os.path.basename(csv_path)}):")
            print(f"  - 记录数: {len(df)}")
            print(f"  - 最小值: {target_values.min():.6f}")
            print(f"  - 最大值: {target_values.max():.6f}")
            print(f"  - 均值: {target_values.mean():.6f}")
            print(f"  - 标准差: {target_values.std():.6f}")
            print(f"  - 中位数: {pd.Series(target_values).median():.6f}")
            
            # 检查是否值太小（可能被过度缩放）
            if target_values.max() < 100:
                print(f"  ⚠️  警告: 目标值太小，可能需要放大1000倍")
            elif target_values.min() > 1000:
                print(f"  ✅ 目标值范围正常")
            else:
                print(f"  ℹ️  目标值范围中等")
                
        else:
            print(f"\n❌ 文件不存在: {csv_path}")

if __name__ == "__main__":
    check_target_values()
