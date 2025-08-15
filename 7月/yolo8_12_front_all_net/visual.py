import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import cv2

# 导入你的自定义模型
from custom_modules.custom_tasks import RegressionModel
IMAGE_NAME = '100359正.jpg'

def generate_gradcam(model, target_layer, image_tensor, image_pil):
    feature_maps = []
    gradients = []

    def forward_hook(module, input, output):
        feature_maps.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    model.eval()
    output = model(image_tensor)
    model.zero_grad()
    output.backward()

    handle_forward.remove()
    handle_backward.remove()

    grads_val = gradients[0].cpu().data.numpy()
    target = feature_maps[0].cpu().data.numpy()[0, :]

    weights = np.mean(grads_val, axis=(2, 3))[0, :]
    cam = np.zeros(target.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * target[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (image_pil.width, image_pil.height))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    original_image = np.array(image_pil)
    superimposed_img = heatmap * 0.4 + original_image * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img, output.item()

# 配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\yolo8_12_front_all_net"
RUNS_DIR = PROJECT_ROOT
IMAGE_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'images', IMAGE_NAME)
YAML_DIR = os.path.join(PROJECT_ROOT, 'yaml')

# 读取图片
img_pil = Image.open(IMAGE_PATH).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)

# 遍历所有 runs-* 文件夹
for folder in os.listdir(RUNS_DIR):
    folder_path = os.path.join(RUNS_DIR, folder)
    if os.path.isdir(folder_path) and folder.startswith('runs'):
        best_pt = os.path.join(folder_path, 'best.pt')
        # 自动推断yaml文件名
        yaml_base = folder.replace('runs-', '')  # 例如 runs-yolov11n-r-att-conv -> yolov11n-r-att-conv
        yaml_path = os.path.join(YAML_DIR, f'{yaml_base}.yaml')
        if not os.path.exists(best_pt):
            print(f"{folder} 没有 best.pt，跳过")
            continue
        if not os.path.exists(yaml_path):
            print(f"{folder} 没有对应的yaml: {yaml_path}，跳过")
            continue
        print(f"处理: {best_pt}，使用yaml: {yaml_path}")

        # 加载自定义模型
        model = RegressionModel(yaml_path, ch=3)
        model.load_state_dict(torch.load(best_pt, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        # 选择目标层（请根据你的模型结构调整）
        target_layer = model.model[-2]

        # 生成Grad-CAM
        heatmap_image, prediction = generate_gradcam(model, target_layer, img_tensor, img_pil)

        # 保存结果
        save_path = os.path.join(folder_path, 'gradcam_' + IMAGE_NAME)
        Image.fromarray(heatmap_image).save(save_path)
        print(f"已保存: {save_path}")