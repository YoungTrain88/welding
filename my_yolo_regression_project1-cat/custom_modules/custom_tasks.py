# custom_modules/custom_tasks.py

import torch
import torch.nn as nn
import yaml
from copy import deepcopy

# 从 ultralytics 导入所有我们需要的模块和函数
from ultralytics.nn.modules import Conv, C2f, Bottleneck, SPPF, Concat
from ultralytics.utils.ops import make_divisible
from ultralytics.nn.modules.block import *

# ==================================================================
# 1. 首先，定义我们的 RegressionHead (保持不变)
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


# ==================================================================
# 2. 接着，定义一个我们自己的、能识别 RegressionHead 的模型解析器
# ==================================================================
def parse_custom_model(d, ch, verbose=True):
    if verbose:
        print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40} {'arguments':<30}")
    
    nc, gd, gw, ch_mul = (d.get(x) for x in ('nc', 'depth_multiple', 'width_multiple', 'ch_multiple'))
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        if m == 'RegressionHead':
            m = RegressionHead
        else:
            m = eval(m) if isinstance(m, str) else m

        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except (NameError, SyntaxError):
                pass
        
        n = max(round(n * gd), 1) if n > 1 else n
        
        # ----------- 核心修改在这里 -----------
        if m in (Conv, C2f, Bottleneck, SPPF, Concat):
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(c2 * gw, ch_mul or 8)
            args = [c1, c2, *args[1:]]
            if m is C2f:
                args.insert(2, n)
                n = 1
        
        # --- 新增的处理分支，专门为 RegressionHead 准备参数 ---
        elif m is RegressionHead:
            c1 = ch[f] # 输入通道数来自上一层
            c2 = 1     # 输出通道数在回归任务中为1
            args = [c1] # 将 c1 作为参数传递给 RegressionHead 的 __init__ 方法
        # ------------------------------------------------
            
        else: # 处理其他类型的模块
            c2 = ch[f]
        # ----------------------------------------

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
# 3. 最后，RegressionModel 的定义保持不变
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