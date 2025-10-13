# my_yolo_regression_project/custom_modules/custom_tasks.py (修正版)

import math
import os

# 从 ultralytics 导入所有我们需要的模块和函数
import sys
from copy import deepcopy

import torch
import torch.nn as nn
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ultralytics.nn.modules import C3, PSA, SPPF, Bottleneck, C2f, C3k2, Concat, Conv, SCDown
from ultralytics.utils.ops import make_divisible


# ==================================================================
# 1. 定义您自定义的新模块 (此处假设您已定义好，或使用下面的占位符)
# ==================================================================
# ... (您的 RegressionHead, AFGCAttention, ARCBlock, CARC 等类的定义放在这里)
# ... (为确保完整性，我将再次提供这些类的正确定义)
class RegressionHead(nn.Module):
    def __init__(self, c1: int):
        super().__init__()
        c_ = 1280
        self.conv = Conv(c1, c_, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.1, inplace=True)
        self.linear = nn.Linear(c_, 1)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x_flat = self.drop(self.pool(self.conv(x)).flatten(1))
        return self.linear(x_flat)


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
def parse_custom_model(d, ch, verbose=True):
    """一个逻辑正确的、健壮的模型解析器。."""
    if verbose:
        print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40} {'arguments':<30}")

    # 获取模型缩放系数
    nc = d.get("nc")
    gd = d.get("depth_multiple", 1.0)  # repeats
    gw = d.get("width_multiple", 1.0)  # channels

    ch = [ch]  # 输入通道
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        # --- 1. 将模块名字符串转换为类 ---
        # 首先在当前文件的全局变量中查找，如果找不到，再去 ultralytics.nn.modules 中查找
        # 这样可以确保优先使用我们自己定义的模块
        m_class = globals().get(m)
        if m_class is None:
            try:
                m_class = getattr(torch.nn, m)
            except AttributeError:
                m_class = getattr(__import__("ultralytics.nn.modules", fromlist=[m]), m)
        m = m_class

        # --- 2. 解析参数 ---
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except (NameError, SyntaxError):
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain

        # --- 3. 正确推断输入/输出通道和参数 ---
        if m in [Conv, C2f, CARC, C3, C3k2, Bottleneck, PSA]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8) if c2 != nc else c2
            args = [c1, c2, *args[1:]]
            if m in [C2f, C3, C3k2]:
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
        elif m is Concat:
            c2 = sum(ch[x] for x in f)

        elif m in [AFGCAttention, PSA]:  # 注意力模块
            c1 = c2 = ch[f]
            args = [c1]

        elif m is RegressionHead:
            c1 = ch[f]
            c2 = 1
            args = [c1]

        else:
            c1 = c2 = ch[f]
            if args:
                c2 = make_divisible(args[0] * gw, 8) if args[0] != nc else args[0]
                args = [c1, c2, *args[1:]]
            else:
                args = [c1]

        # --- 4. 创建模块实例 ---
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        if verbose:
            print(f"{i:>3}{str(f):>18}{n:>3}{np:10.0f}  {t:<40} {str(args):<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


# ==================================================================
# 3. RegressionModel 的定义 (保持不变)
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

    def forward(self, x):
        return self.model(x)

    def initialize_biases(self, cf=None):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "bn") and isinstance(m.bn, nn.BatchNorm2d):
                    with torch.no_grad():
                        m.bn.bias.zero_()
