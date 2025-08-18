from typing import List, Optional, Union

import torch
import torch.nn as nn


# 假设 Conv 类已经定义好，这里我们用一个标准的 Conv2d 替代作为示例
# from ultralytics.nn.modules import Conv  # 在实际项目中，你应该从原处导入
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        if p is None:
            p = k // 2  # 自动填充，保证输出尺寸不变
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ==================== 修改后的回归头 ====================


class RegressionHead(nn.Module):
    """YOLO Regression head, to transform feature maps x(b, c1, h, w) to a single regression value x(b, 1)."""

    export = False  # export mode

    def __init__(self, c1: int, k: int = 1, s: int = 1, p: Optional[int] = None, g: int = 1):
        """
        Initialize YOLO regression head.

        Args:
            c1 (int): Number of input channels.
            k (int, optional): Kernel size for the initial convolution.
            s (int, optional): Stride for the initial convolution.
            p (int, optional): Padding for the initial convolution.
            g (int, optional): Groups for the initial convolution.
        """
        super().__init__()
        c_ = 1280  # Intermediate feature size, can be tuned
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling to get (b, c_, 1, 1)
        self.drop = nn.Dropout(p=0.1, inplace=True)  # Dropout can be adjusted or removed

        # --- KEY CHANGE 1: The output dimension of the linear layer is 1 ---
        self.linear = nn.Linear(c_, 1)  # Output a single value for regression

    def forward(self, x: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Perform a forward pass. The output is the raw regression value."""
        if isinstance(x, list):
            x = torch.cat(x, 1)

        # Forward pass through conv, pool, flatten, and dropout
        x_flat = self.drop(self.pool(self.conv(x)).flatten(1))

        # Get the final regression value from the linear layer
        prediction = self.linear(x_flat)

        # --- KEY CHANGE 2: No softmax, always return the raw prediction value ---
        return prediction


# ...existing code...

if __name__ == "__main__":
    # 假设输入特征图的通道数为 256
    c1 = 256
    model = RegressionHead(c1)

    # 构造一个模拟输入，batch size=4，通道数=256，空间尺寸为32x32
    x = torch.randn(4, c1, 32, 32)

    # 前向推理
    output = model(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", output.shape)
    print("输出内容:", output)
# ...existing code...
