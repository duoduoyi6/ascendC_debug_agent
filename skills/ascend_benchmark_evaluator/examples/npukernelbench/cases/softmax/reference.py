import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Softmax 归一化模型
    """
    def __init__(self, dim: int = -1):
        super(Model, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.dim)


def get_inputs():
    """生成输入"""
    return [torch.randn(128, 64, dtype=torch.float32)]


def get_init_inputs():
    """返回初始化参数"""
    return [-1]
