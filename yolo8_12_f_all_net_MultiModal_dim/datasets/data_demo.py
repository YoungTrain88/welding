import pandas as pd
import numpy as np

# 读取原始train.csv
df = pd.read_csv(r'C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\yolo8-12-正反拼接300轮 -多模态\datasets\val.csv', encoding='utf-8')

# 生成两列随机特征
num_samples = len(df)
df['feature1'] = np.random.rand(num_samples)
df['feature2'] = np.random.rand(num_samples)

# 保存为新文件（或覆盖原文件）
df.to_csv('train_with_features.csv', index=False, encoding='utf-8')
print(df.head())