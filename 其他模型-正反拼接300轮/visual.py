import os

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import models
from torchvision.transforms import transforms

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = r"C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\其他模型-正反拼接300轮"
RUNS_DIR = PROJECT_ROOT
IMAGE_NAME = "100359正_拼接.jpg"
IMAGE_PATH = os.path.join(PROJECT_ROOT, "datasets", "images", IMAGE_NAME)


def get_target_layer(model, model_name):
    if "resnet" in model_name:
        return model.layer4[-1]
    elif "densenet" in model_name:
        return model.features[-1]
    elif "vgg" in model_name:
        return model.features[-1]
    elif "mobilenet" in model_name:
        return model.features[-1]
    elif "efficientnet" in model_name or "convnext" in model_name or "resnext" in model_name:
        if hasattr(model, "blocks"):
            return model.blocks[-1]
        elif hasattr(model, "stages"):
            return model.stages[-1]
        elif hasattr(model, "layers"):
            return model.layers[-1]
        else:
            return list(model.children())[-2]
    elif "vit" in model_name or "swin" in model_name:
        # ViT/Swin不需要target_layer
        return None
    else:
        raise ValueError(f"未知模型结构: {model_name}")


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

    if grads_val.ndim != 4 or target.ndim != 3:
        raise RuntimeError(f"Grad-CAM只支持4维特征图，当前grad shape: {grads_val.shape}, fmap shape: {target.shape}")

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


def attention_rollout(model, image_tensor, image_pil, model_name):
    model.eval()
    attn_weights = []
    hooks = []

    def save_attn(module, input, output):
        attn_weights.append(output.detach())

    # ViT
    if "vit" in model_name:
        for blk in model.blocks:
            # timm标准ViT为blk.attn.proj_drop
            if hasattr(blk.attn, "proj_drop"):
                hooks.append(blk.attn.proj_drop.register_forward_hook(save_attn))
            elif hasattr(blk.attn, "attn_drop"):
                hooks.append(blk.attn.attn_drop.register_forward_hook(save_attn))
    # Swin
    elif "swin" in model_name:
        # Swin最后一层可能没有blocks
        if hasattr(model.layers[-1], "blocks"):
            for blk in model.layers[-1].blocks:
                if hasattr(blk.attn, "proj_drop"):
                    hooks.append(blk.attn.proj_drop.register_forward_hook(save_attn))
                elif hasattr(blk.attn, "attn_drop"):
                    hooks.append(blk.attn.attn_drop.register_forward_hook(save_attn))
    else:
        raise RuntimeError("Attention Rollout仅支持ViT/Swin")
    # 前向
    with torch.no_grad():
        output = model(image_tensor)
    for h in hooks:
        h.remove()
    if not attn_weights:
        raise RuntimeError("未捕获到注意力权重，模型结构可能不兼容。")
    attn_mat = [a.cpu().numpy() for a in attn_weights]
    attn_mat = [np.mean(a, axis=1) for a in attn_mat]
    # 只取[CLS] token对所有patch的注意力
    result = np.eye(attn_mat[0].shape[-1])
    for a in attn_mat:
        # 兼容不同维度
        if a.ndim == 3:
            a = a[:, 0, :]  # 只取CLS
        elif a.ndim == 2:
            a = a[0, :]  # 只有一个batch
        result = np.matmul(a, result)
    if result.shape[0] > 1:
        mask = result[0, 1:]  # 去掉CLS本身
    else:
        mask = result[0, 1:]
    num_patches = int(np.sqrt(mask.shape[0]))
    if num_patches * num_patches != mask.shape[0]:
        raise RuntimeError(f"注意力mask无法reshape为正方形: {mask.shape[0]}")
    mask = mask.reshape(num_patches, num_patches)
    mask = cv2.resize(mask, (image_pil.width, image_pil.height))
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    original_image = np.array(image_pil)
    superimposed_img = heatmap * 0.4 + original_image * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img, output.item()


preprocess = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img_pil = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)

for folder in os.listdir(RUNS_DIR):
    folder_path = os.path.join(RUNS_DIR, folder)
    if os.path.isdir(folder_path) and folder.startswith("runs_"):
        best_pt = os.path.join(folder_path, "best.pt")
        if not os.path.exists(best_pt):
            print(f"{folder} 没有 best.pt，跳过")
            continue
        model_name = folder.replace("runs_", "")
        print(f"处理: {best_pt}，模型: {model_name}")

        # 加载模型结构
        try:
            if model_name == "resnet50":
                model = models.resnet50(weights=None)
                num_ftrs = model.fc.in_features
                model.fc = torch.nn.Linear(num_ftrs, 1)
            elif model_name == "vgg16":
                model = models.vgg16(weights=None)
                num_ftrs = model.classifier[6].in_features
                model.classifier[6] = torch.nn.Linear(num_ftrs, 1)
            elif model_name == "densenet121":
                model = models.densenet121(weights=None)
                num_ftrs = model.classifier.in_features
                model.classifier = torch.nn.Linear(num_ftrs, 1)
            elif model_name == "mobilenet_v3_large":
                model = models.mobilenet_v3_large(weights=None)
                num_ftrs = model.classifier[3].in_features
                model.classifier[3] = torch.nn.Linear(num_ftrs, 1)
            else:
                model = timm.create_model(model_name, pretrained=False, num_classes=1)
        except Exception as e:
            print(f"模型 {model_name} 加载失败: {e}")
            continue

        model.load_state_dict(torch.load(best_pt, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        # 判断模型类型
        try:
            if "vit" in model_name or "swin" in model_name:
                # Attention Rollout
                try:
                    heatmap_image, prediction = attention_rollout(model, img_tensor, img_pil, model_name)
                    save_path = os.path.join(folder_path, "attnrollout_" + IMAGE_NAME)
                    Image.fromarray(heatmap_image).save(save_path)
                    print(f"已保存: {save_path}，预测值: {prediction:.4f}")
                except Exception as e:
                    print(f"模型 {model_name} Attention Rollout 失败: {e}")
            else:
                # Grad-CAM
                try:
                    target_layer = get_target_layer(model, model_name)
                    heatmap_image, prediction = generate_gradcam(model, target_layer, img_tensor, img_pil)
                    save_path = os.path.join(folder_path, "gradcam_" + IMAGE_NAME)
                    Image.fromarray(heatmap_image).save(save_path)
                    print(f"已保存: {save_path}，预测值: {prediction:.4f}")
                except Exception as e:
                    print(f"模型 {model_name} Grad-CAM 失败: {e}")
        except Exception as e:
            print(f"模型 {model_name} 可视化失败: {e}")
