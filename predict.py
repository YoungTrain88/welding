import torch
from torchvision import transforms
from PIL import Image

# 导入你的自定义模型
from my_yolo_r_p1_c_s_att_conv.custom_modules.custom_tasks import RegressionModel

# 配置
MODEL_YAML_PATH = r'C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\my_yolo_r_p1_c_s_att_conv\yolov11n-r-att-conv.yaml'
WEIGHTS_PATH = r'C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\my_yolo_r_p1_c_s_att_conv\runs-yolo11n-AFAR\best.pt'  # 你的训练好的pt文件
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 加载模型结构
model = RegressionModel(MODEL_YAML_PATH, ch=3).to(DEVICE)
model.eval()

# 2. 加载权重
ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
if 'model' in ckpt:
    state_dict = ckpt['model'].float().state_dict()
else:
    state_dict = ckpt
model.load_state_dict(state_dict, strict=False)

# 3. 定义与训练一致的预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 替换为你的IMG_SIZE
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. 读取图片并预处理
img_path = r'C:\Users\User\Desktop\焊接\ultralytics-main\ultralytics-main\my_yolo_r_p1_c_s_att_conv\datasets\images\101678正_拼接.jpg'  # 替换为你要预测的图片路径
img = Image.open(img_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0).to(DEVICE)

# 5. 推理
with torch.no_grad():
    pred = model(input_tensor)
print('预测值:', pred.cpu().numpy())