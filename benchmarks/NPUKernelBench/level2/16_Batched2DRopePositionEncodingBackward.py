import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Batched 2D RoPE Position Encoding Backward Module.

    Computes the backward pass (gradient) for batched 2D Rotary Position Encoding.
    Uses chain rule to compute gradient w.r.t. idx_theta:
    - d(cos(x))/dx = -sin(x)
    - d(sin(x))/dx = cos(x)
    Therefore: grad_idx_theta = -grad_cos * sin(idx_theta) + grad_sin * cos(idx_theta)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_cos: torch.Tensor, grad_sin: torch.Tensor, idx_theta: torch.Tensor) -> torch.Tensor:
        """
        Backward pass for batched 2D RoPE position encoding.

        Args:
            grad_cos: Gradient w.r.t. cos output [batch_size, seq_len, head_dim]
            grad_sin: Gradient w.r.t. sin output [batch_size, seq_len, head_dim]
            idx_theta: Saved angles from forward pass [batch_size, seq_len, head_dim]

        Returns:
            grad_idx_theta: Gradient w.r.t. idx_theta [batch_size, seq_len, head_dim]
        """
        sin_theta = torch.sin(idx_theta)
        cos_theta = torch.cos(idx_theta)
        grad_cos_f32 = grad_cos.to(torch.float32)
        grad_sin_f32 = grad_sin.to(torch.float32)
        grad_idx_theta = -grad_cos_f32 * sin_theta + grad_sin_f32 * cos_theta
        return grad_idx_theta
