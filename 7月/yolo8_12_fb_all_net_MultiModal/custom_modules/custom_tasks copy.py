# my_yolo_regression_project/custom_modules/custom_tasks.py (修正版)

import math
import os

# 从 ultralytics 导入所有我们需要的模块和函数
import sys
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ultralytics.nn.modules import (
    C2PSA,
    C3,
    PSA,
    SPPF,
    A2C2f,
    Bottleneck,
    C2f,
    C2fCIB,
    C3k2,
    Concat,
    Conv,
    SCDown,
)
from ultralytics.utils.ops import make_divisible

# ==================================================================
# 1. 定义您自定义的新模块 (此处假设您已定义好，或使用下面的占位符)
# ==================================================================
# ... (您的 RegressionHead, AFGCAttention, ARCBlock, CARC 等类的定义放在这里)
# ... (为确保完整性，我将再次提供这些类的正确定义)
# class RegressionHead(nn.Module):
#     def __init__(self, c1: int):
#         super().__init__()
#         c_ = 1280
#         self.conv = Conv(c1, c_, 1, 1)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.drop = nn.Dropout(p=0.1, inplace=True)
#         self.linear = nn.Linear(c_, 1)
#     def forward(self, x):
#         if isinstance(x, list): x = torch.cat(x, 1)
#         x_flat = self.drop(self.pool(self.conv(x)).flatten(1))
#         return self.linear(x_flat)


class AFGCAttention(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super().__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ARCBlock(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 1)
        self.cv2 = Conv(c2, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class CARC(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(ARCBlock(c_, c_) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


# --- 其他您可能用到的自定义模块的占位符 ---
# 如果您使用了其他未在ultralytics.nn.modules中定义的模块，需要在此处添加它们的定义
# 例如 C3k2, C2PSA, SCDown, PSA, ABlock, C3k, A2C2f
# class C3k2(C3): pass # 示例，假设它与C3类似
# class C2PSA(nn.Module): pass # 占位符
# class SCDown(nn.Module): pass # 占位符
# class PSA(nn.Module): pass # 占位符
# class ABlock(nn.Module): pass # 占位符
# class C3k(C3): pass # 占位符
# class A2C2f(C2f): pass # 占位符


# ==================================================================
# 2. 修正后的模型解析器
# ==================================================================
# 在 custom_tasks.py 中，用这个函数替换掉旧的 parse_custom_model
def parse_custom_model(d, ch, verbose=True):
    if verbose:
        print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40} {'arguments':<30}")

    nc = d.get("nc")
    if not isinstance(nc, int):
        raise TypeError(
            f"YAML配置错误: 'nc' (类别/输出数量) 必须在YAML文件的顶部定义为一个整数，但现在的值是 {nc}。请在你的 .yaml 文件顶部添加 'nc: 1'。"
        )

    gd = d.get("depth_multiple", 1.0)
    gw = d.get("width_multiple", 1.0)

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        # >>> 调试代码可以暂时保留或移除
        # print(f">>> 正在解析第 {i} 层: 模块={m}, 参数={args}")

        # ==================== 从这里开始是全新的、更健壮的解析逻辑 ====================
        m_str = m if isinstance(m, str) else m.__name__
        m_class = None

        try:
            # 优先在当前文件的全局变量中查找 (我们自己定义的模块，如 RegressionHead)
            m_class = globals().get(m_str)

            if m_class is None:
                # 其次，在 torch.nn 中查找
                m_class = getattr(torch.nn, m_str)

        except (AttributeError, TypeError):
            # 最后，在 ultralytics.nn.modules 中查找
            try:
                m_class = getattr(__import__("ultralytics.nn.modules", fromlist=[m_str]), m_str)
            except (ImportError, AttributeError):
                raise NameError(f"无法在任何已知位置解析模块 '{m_str}'。请检查拼写或导入。")

        m = m_class
        # ============================== 新逻辑结束 ==============================

        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except (NameError, SyntaxError):
                pass

        n = max(round(n * gd), 1) if n > 1 else n

        if m in [Conv, C2f, CARC, C3, C3k2, Bottleneck, PSA, A2C2f, C2fCIB]:
            c1 = ch[f]
            if m is C2fCIB:
                c2, args[0] = args[1], c1
            else:
                c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(c2 * gw, 8)
            if m is C2fCIB:
                args[1] = c2
            else:
                args = [c1, c2, *args[1:]]
            if m in [C2f, C3, C3k2, A2C2f, C2fCIB]:
                args.insert(2, n)
                n = 1
        elif m is SPPF:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8) if c2 != nc else c2
            args = [c1, c2, *args[1:]]
        elif m is SCDown:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8) if c2 != nc else c2
            args = [c1, c2, *args[1:]]
        elif m is CustomConcat or m is Concat:  # 同时兼容 Concat 和我们自己的 CustomConcat
            c2 = sum(ch[x] for x in f)
        elif m in [torch.nn.Upsample, nn.MaxPool2d]:  # 使用 torch.nn.Upsample
            c2 = ch[f]
        elif m in [AFGCAttention, PSA, C2PSA]:
            c1 = c2 = ch[f]
            args = [c1]
        elif m is RegressionHead:
            try:
                c1 = sum(ch[x] for x in f) if isinstance(f, list) else ch[f]
            except IndexError:
                raise IndexError(
                    f"\n\n[YAML配置错误] 在解析层 {i} ({m.__name__}) 时出错:\n  > 'from' 字段指定了无效的来源层索引: {f}\n"
                )
            c2 = 1
            args = [c1]
        else:
            c1 = ch[f]
            if args:
                if isinstance(args[0], int):
                    c2 = make_divisible(args[0] * gw, 8) if args[0] != nc else args[0]
                    args = [c1, c2, *args[1:]]
                else:
                    c2 = c1
            else:
                c2 = c1
                args = [c1]

        if m is CustomConcat:
            m_ = m(*args)
        elif m is Concat:
            m_ = m(dimension=args[0])
        else:
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)

        t = str(m).split(".")[-1].replace("'>", "")
        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        if verbose:
            print(f"{i:>3}{str(f):>18}{n:>3}{np:10.0f}  {t:<40} {str(args):<30}")

        save.extend(x for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.ModuleList(layers), sorted(list(set(save)))


# my_yolo_regression_project/custom_modules/custom_tasks.py (多模态修改版)
# ... (文件顶部的所有 import 和自定义模块 AFGCAttention, ARCBlock, CARC, CustomConcat 保持不变) ...


class CustomConcat(nn.Module):
    # ... 此模块代码完全不变 ...
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        target_size = x[0].shape[2:]
        x = [
            F.interpolate(img, size=target_size, mode="bilinear", align_corners=False)
            if img.shape[2:] != target_size
            else img
            for img in x
        ]
        return torch.cat(x, self.d)


class RegressionHead(nn.Module):
    def __init__(self, c1: int):
        super().__init__()
        c_ = 1280  # 这个值很重要，它是图像特征的维度
        self.c_ = c_
        self.conv = Conv(c1, c_, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.1, inplace=True)
        self.linear = nn.Linear(c_, 1)

    def forward(self, x, return_features=False):  # <--- 修改点 1: 增加参数
        if isinstance(x, list):
            target_size = x[0].shape[2:]
            x = [
                F.interpolate(img, size=target_size, mode="bilinear", align_corners=False)
                if img.shape[2:] != target_size
                else img
                for img in x
            ]
            x = torch.cat(x, 1)

        features_flat = self.drop(self.pool(self.conv(x)).flatten(1))

        if return_features:  # <--- 修改点 2: 根据参数返回特征或最终结果
            return features_flat

        return self.linear(features_flat)


# ==================================================================
# 3. RegressionModel 的定义 (保持不变)
# ==================================================================
# ==================================================================
# 3. RegressionModel 的定义 (采用官方标准 forward 逻辑)
# ==================================================================
class RegressionModel(nn.Module):
    def __init__(self, cfg, ch=3, nc=None, verbose=True):
        super().__init__()
        if isinstance(cfg, str):
            with open(cfg, encoding="utf-8") as f:
                self.yaml = yaml.safe_load(f)
        else:
            self.yaml = cfg

        self.model, self.save = parse_custom_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.initialize_biases()

    def forward(self, x, return_features=False):  # <--- 修改点 3: 增加参数
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            # <--- 修改点 4: 拦截并处理 RegressionHead
            if isinstance(m, RegressionHead) and return_features:
                return m(x, return_features=True)

            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def initialize_biases(self, cf=None):
        # ... 此方法代码完全不变 ...
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "bn") and isinstance(m.bn, nn.BatchNorm2d):
                    with torch.no_grad():
                        m.bn.bias.zero_()
