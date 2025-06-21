import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import transforms

# 存储特征图和梯度的全局变量
feature_maps = []
gradients = []

def save_gradient(grad):
    """一个hook函数，用于在反向传播时保存梯度"""
    gradients.append(grad)

def generate_gradcam(model, target_layer, image_tensor, image_pil):
    """
    为回归模型生成Grad-CAM热力图。

    Args:
        model (nn.Module): 训练好的模型。
        target_layer (nn.Module): 我们希望可视化的目标卷积层。
        image_tensor (torch.Tensor): 经过预处理、准备输入模型的图片张量。
        image_pil (PIL.Image): 原始的、未经处理的PIL图片，用于叠加显示。

    Returns:
        numpy.ndarray: 叠加了热力图的图片。
    """
    global feature_maps, gradients
    feature_maps = []
    gradients = []

    # 1. 注册hook
    # a. 前向hook，用于捕获目标层的输出特征图
    def forward_hook(module, input, output):
        feature_maps.append(output)

    # b. 反向hook，用于捕获目标层输出的梯度
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
        
    # 将hook挂载到目标层
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    # 2. 前向传播
    model.eval()
    output = model(image_tensor) # 得到回归预测值

    # 3. 反向传播
    model.zero_grad()
    # 这是与分类任务的关键区别：我们直接对标量输出进行反向传播
    output.backward()

    # 4. 移除hook，防止内存泄漏
    handle_forward.remove()
    handle_backward.remove()

    # 5. 计算热力图
    # 获取目标层的梯度和特征图
    grads_val = gradients[0].cpu().data.numpy()
    target = feature_maps[0].cpu().data.numpy()[0, :]

    # 计算每个通道的权重 (alpha)
    weights = np.mean(grads_val, axis=(2, 3))[0, :]
    cam = np.zeros(target.shape[1:], dtype=np.float32)

    # 计算特征图的加权和
    for i, w in enumerate(weights):
        cam += w * target[i, :, :]

    # ReLU和归一化
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (image_pil.width, image_pil.height))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # 6. 可视化
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    original_image = np.array(image_pil)
    superimposed_img = heatmap * 0.4 + original_image * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img, output.item()

# visualize.py
import torch
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

# 假设上面的 Grad-CAM 代码和下面的代码在同一个文件中
# from train_resnet import create_resnet50_regression # 或者从您的训练脚本导入模型创建函数

# --- 1. 定义模型创建函数 (与训练时一致) ---
import torchvision.models as models
import torch.nn as nn
def create_resnet50_regression():
    model = models.resnet50(weights=None) # 加载模型结构，但不加载预训练权重
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model


# --- 2. 加载您训练好的模型 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_WEIGHTS_PATH = r'C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\my_yolo_regression_project1-cat-shuffed\runs_resnet50\best.pt' # ❗️❗️ 修改为您最佳模型的路径
IMAGE_PATH = r'C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\my_yolo_regression_project1-cat-shuffed\datasets\images\100377正_拼接.jpg' # ❗️❗️ 修改为您想测试的一张图片路径

# 创建模型实例并加载权重
model = create_resnet50_regression()
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- 3. 确定目标层 ---
# 对于ResNet50，最后一个卷积块是 model.layer4
target_layer = model.layer4[-1] # 选择layer4的最后一个Bottleneck块

# --- 4. 准备输入图片 ---
img_pil = Image.open(IMAGE_PATH).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)

# --- 5. 生成并显示热力图 ---
heatmap_image, prediction = generate_gradcam(model, target_layer, img_tensor, img_pil)

# 使用 matplotlib 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_pil)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(heatmap_image)
plt.title(f"Grad-CAM Heatmap\nPredicted Value: {prediction:.2f}")
plt.axis('off')

plt.show()