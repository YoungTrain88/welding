# my_yolo_regression_project/custom_modules/custom_tasks.py (最终修正版)
# 请用此文件的全部内容替换你现有的 custom_tasks.py 文件

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# 确保在你的环境中可以找到 ultralytics 包
try:
    from ultralytics.nn.modules import (
        C2PSA,
        C3,
        PSA,
        SPPF,
        A2C2f,
        ABlock,
        Bottleneck,
        C2f,
        C2fCIB,
        C3k2,
        Concat,
        Conv,
        DWConv,
        SCDown,
    )
    from ultralytics.utils.ops import make_divisible
except ImportError:
    raise ImportError("请确保 ultralytics 包已正确安装，并且版本支持 YOLOv10。")


# ==================================================================
# 1. 您的自定义模块定义 (新增 CustomConcat)
# ==================================================================


class CustomConcat(nn.Module):
    """一个“聪明的”拼接模块，它在拼接前会自动统一输入特征图的空间尺寸。."""

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        # x 是一个张量列表
        # 目标尺寸是列表中第一个（通常是最大的）特征图的尺寸
        target_size = x[0].shape[2:]

        # 如果其他特征图的尺寸与目标尺寸不匹配，就进行上采样
        x = [
            F.interpolate(img, size=target_size, mode="bilinear", align_corners=False)
            if img.shape[2:] != target_size
            else img
            for img in x
        ]

        # 现在所有张量尺寸都已统一，可以安全地进行拼接
        return torch.cat(x, self.d)


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
            # RegressionHead 也使用我们新的 CustomConcat 逻辑来确保尺寸一致
            target_size = x[0].shape[2:]
            x = [
                F.interpolate(img, size=target_size, mode="bilinear", align_corners=False)
                if img.shape[2:] != target_size
                else img
                for img in x
            ]
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


# ==================================================================
# 2. 模型解析器 (使用 CustomConcat)
# ==================================================================
def parse_custom_model(d, ch, verbose=True):
    if verbose:
        print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40} {'arguments':<30}")

    nc = d.get("nc")
    gd = d.get("depth_multiple", 1.0)
    gw = d.get("width_multiple", 1.0)

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m_class = None
        if isinstance(m, str):
            if m.startswith("nn."):
                module_name = m[3:]
                try:
                    m_class = getattr(nn, module_name)
                except AttributeError:
                    raise AttributeError(f"模块 '{module_name}' 在 torch.nn 中未找到。")
            else:
                module_name = m
                # ==================== 关键修改点 1 ====================
                # 当YAML中模块名为 Concat 时, 强制替换为我们自己的 CustomConcat
                if module_name == "Concat":
                    m_class = CustomConcat
                else:
                    m_class = globals().get(module_name)
                # =====================================================
                if m_class is None:
                    try:
                        m_class = getattr(nn, module_name)
                    except AttributeError:
                        pass
                if m_class is None:
                    try:
                        m_class = getattr(__import__("ultralytics.nn.modules", fromlist=[module_name]), module_name)
                    except (ImportError, AttributeError):
                        pass
        else:
            m_class = m

        if m_class is None:
            raise NameError(f"无法解析模块 '{m}'。")
        m = m_class

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
        elif m is CustomConcat:
            c2 = sum(ch[x] for x in f)
        elif m in [nn.Upsample, nn.MaxPool2d]:
            c2 = ch[f]
        elif m in [AFGCAttention, PSA]:
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

        # ==================== 关键修改点 2 ====================
        # CustomConcat 的参数在 YAML 中定义 (通常是 [1])
        if m is CustomConcat:
            m_ = m(*args)
        else:
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        # =====================================================

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

    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def initialize_biases(self, cf=None):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "bn") and isinstance(m.bn, nn.BatchNorm2d):
                    with torch.no_grad():
                        m.bn.bias.zero_()
