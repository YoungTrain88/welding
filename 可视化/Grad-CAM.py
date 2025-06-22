import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import cv2

# 导入你的自定义模型
from my_yolo_r_p1_c_s_att_conv.custom_modules.custom_tasks import RegressionModel

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

# 配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_ROOT, '..', 'yolo8-12正反拼接-300轮')
IMAGE_PATH = os.path.join(PROJECT_ROOT, '..', 'my_yolo_regression_project1-cat-shuffed', 'datasets', 'images', '100359正_拼接.jpg')
YAML_PATH = os.path.join(PROJECT_ROOT, '..', 'my_yolo_r_p1_c_s_att_conv', 'yoloV11n-r-att-conv.yaml')  # 你的yaml结构文件

# 遍历所有 runs-* 文件夹
for folder in os.listdir(RUNS_DIR):
    folder_path = os.path.join(RUNS_DIR, folder)
    if os.path.isdir(folder_path) and folder.startswith('runs'):
        best_pt = os.path.join(folder_path, 'best.pt')
        if not os.path.exists(best_pt):
            print(f"{folder} 没有 best.pt，跳过")
            continue
        print(f"处理: {best_pt}")

        # 加载自定义模型
        model = RegressionModel(YAML_PATH, ch=3)
        model.load_state_dict(torch.load(best_pt, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        # 选择目标层（假设最后一个卷积层叫 model.model[-2]，请根据你的模型结构调整）
        # 你可以 print(model) 查看结构，选择合适的层
        target_layer = model.model[-2]  # 这里请根据你的RegressionModel实际结构调整

        # 读取图片
        img_pil = Image.open(IMAGE_PATH).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)

        # 生成Grad-CAM
        heatmap_image, prediction = generate_gradcam(model, target_layer, img_tensor, img_pil)

        # 保存结果
        save_path = os.path.join(folder_path, 'gradcam_100359正_拼接.png')
        Image.fromarray(heatmap_image).save(save_path)
        print(f"已保存: {save_path}")