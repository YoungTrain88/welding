import os
import csv
from PIL import Image
import random

# --- 配置 ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT_DIR, 'datasets')
IMAGE_DIR = os.path.join(DATASET_DIR, 'images')
NUM_IMAGES = 10
TRAIN_SPLIT = 0.8 # 80% 的数据用于训练

def create_dummy_dataset():
    """
    自动创建虚拟图片、CSV标签文件和 data.yaml 配置文件。
    """
    print("开始创建虚拟数据集...")

    # 1. 创建目录
    os.makedirs(IMAGE_DIR, exist_ok=True)
    print(f"目录 '{IMAGE_DIR}' 已创建或已存在。")

    # 2. 生成虚拟图片和收集标签信息
    image_paths_and_values = []
    for i in range(NUM_IMAGES):
        # 创建一张简单的彩色图片
        img_filename = f'img_{i:03d}.jpg'
        img_filepath = os.path.join(IMAGE_DIR, img_filename)
        
        # 图片颜色随机，模拟不同的图片内容
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = Image.new('RGB', (200, 200), color=color)
        img.save(img_filepath)

        # 为每张图片生成一个随机的回归目标值
        regression_value = round(random.uniform(5.0, 50.0), 2)
        
        # 存储相对路径和值，以便写入CSV
        relative_path = os.path.join('datasets', 'images', img_filename)
        image_paths_and_values.append((relative_path, regression_value))

    print(f"已成功生成 {NUM_IMAGES} 张虚拟图片。")

    # 3. 分割训练集和验证集
    random.shuffle(image_paths_and_values)
    split_index = int(len(image_paths_and_values) * TRAIN_SPLIT)
    train_data = image_paths_and_values[:split_index]
    val_data = image_paths_and_values[split_index:]

    # 4. 写入 train.csv 和 val.csv 文件
    header = ['image_path', 'target_value']
    
    # 写入 train.csv
    train_csv_path = os.path.join(DATASET_DIR, 'train.csv')
    with open(train_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(train_data)
    print(f"已创建 train.csv，包含 {len(train_data)} 条记录。")

    # 写入 val.csv
    val_csv_path = os.path.join(DATASET_DIR, 'val.csv')
    with open(val_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(val_data)
    print(f"已创建 val.csv，包含 {len(val_data)} 条记录。")


    # 5. 创建 regression_data.yaml 文件
    yaml_path = os.path.join(ROOT_DIR, 'regression_data.yaml')
    yaml_content = f"""
# 数据集配置文件
# 路径相对于项目根目录

train: {os.path.join('datasets', 'train.csv')}
val: {os.path.join('datasets', 'val.csv')}

# 类别名称 (对于回归任务，这里只是一个占位符)
names:
  0: 'value'

"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"已创建 regression_data.yaml。")
    print("\n数据集准备完成！")

if __name__ == '__main__':
    create_dummy_dataset()