# my_yolo_regression_project/custom_modules/custom_tasks.py

import torch
import torch.nn as nn
import yaml
import math
from copy import deepcopy

# 从 ultralytics 导入所有我们需要的模块和函数
from ultralytics.nn.modules import Conv, C2f, Bottleneck, SPPF, Concat,C3k2,C2PSA, ABlock, C3k
from ultralytics.utils.ops import make_divisible


# ==================================================================
# 1. 定义您自定义的新模块 (已补全和修正)
# ==================================================================
class RegressionHead(nn.Module):
    """YOLO Regression head, x(b, c1, h, w) to x(b, 1)."""
    def __init__(self, c1: int, k: int = 1, s: int = 1, p: int = None, g: int = 1):
        super().__init__()
        c_ = 1280
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.1, inplace=True)
        self.linear = nn.Linear(c_, 1)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x_flat = self.drop(self.pool(self.conv(x)).flatten(1))
        return self.linear(x_flat)
class AFGCAttention(nn.Module):
    """
    一个修正后可运行的自适应细粒度通道注意力模块。
    它接收一个输入张量，返回一个经过通道注意力加权后的张量，尺寸不变。
    """
    def __init__(self, channel, b=1, gamma=2):
        super(AFGCAttention, self).__init__()
        # 根据ECA-Net的思想计算一维卷积的核大小
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 对输入特征图进行全局平均池化，得到通道描述符
        y = self.avg_pool(x)
        # Reshape并进行一维卷积来捕获通道间的依赖关系
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # 通过sigmoid获得0-1之间的注意力权重
        y = self.sigmoid(y)
        # 将权重乘以原始输入特征图
        return x * y.expand_as(x)

class ARCBlock(nn.Module):
    """
    一个为CARC模块设计的合理的基础块，包含两个卷积层和一个残差连接。
    """
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 1)
        self.cv2 = Conv(c2, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class CARC(nn.Module):
    """
    Context Aggregation and Refinement Block for CARC-Net
    一个修正后可运行的CARC模块。
    """
    def __init__(self, c1, c2, n=1, e=0.5):  # ch_in, ch_out, number, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(ARCBlock(c_, c_) for _ in range(n)))

    def forward(self, x):
        # 将通过主分支(带ARCBlock)和旁路分支的特征进行拼接
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

# ==================================================================
# 2. 更新模型解析器，让它认识新模块
# ==================================================================
def parse_custom_model(d, ch, verbose=True):
    if verbose:
        print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40} {'arguments':<30}")
    gd = d.get('depth_multiple') or 1.0
    gw = d.get('width_multiple') or 1.0
    ch_mul = d.get('ch_multiple') or 1
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        # 1. 字符串转类
        if isinstance(m, str):
            if m == 'RegressionHead':
                m = RegressionHead
            elif m == 'CARC':
                m = CARC
            elif m == 'AFGCAttention':
                m = AFGCAttention
            else:
                m = eval(m)
        # 2. 参数解析
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except Exception:
                pass
        n = max(round(n * gd), 1) if n > 1 else n

        # 3. 通道推断和参数准备
        if m in (Conv, C2f, Bottleneck, SPPF, Concat, C3k2, C2PSA):
            c1 = ch[f] if isinstance(f, int) else ch[-1]
            c2 = args[0]
            args = [c1, c2, *args[1:]]
            if m is Concat:
                c2 = sum([ch[x] for x in f]) if isinstance(f, (list, tuple)) else ch[f]
                args = [1]  # dim=1
            if m is C2f:
                args.insert(2, n)
                n = 1
        elif m is CARC:
            c1 = ch[f] if isinstance(f, int) else ch[-1]
            c2 = args[0]
            args = [c1, c2, *args[1:]]
        elif m is AFGCAttention:
            c1 = ch[f] if isinstance(f, int) else ch[-1]
            c2 = c1
            args = [c1]
        elif m is RegressionHead:
            c1 = ch[f] if isinstance(f, int) else ch[-1]
            c2 = 1
            args = [c1]
        else:
            c2 = ch[f] if isinstance(f, int) else ch[-1]

    # 4. 实例化
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        if verbose:
            print(f'{i:>3}{str(f):>18}{n:>3}{np:10.0f}  {t:<40} {str(args):<30}')
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

# ==================================================================
# 3. RegressionModel 的定义保持不变
# ==================================================================
class RegressionModel(nn.Module):
    def __init__(self, cfg, ch=3, nc=None, verbose=True):
        super().__init__()
        if isinstance(cfg, str):
            with open(cfg, encoding='utf-8') as f:
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
                if hasattr(m, 'bn') and isinstance(m.bn, nn.BatchNorm2d):
                    with torch.no_grad():
                        m.bn.bias.zero_()

class A2C2f(nn.Module):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """
        Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y