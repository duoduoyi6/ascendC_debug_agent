import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused Add + RMSNorm operation. Computes residual addition followed by Root Mean Square Normalization.
    Formula: x = hidden_states + residual; y = (x * rsqrt(mean(x^2) + eps)) * weight
    Used in models like Qwen3-30B-A3B, Llama-3.1-8B, DeepSeek-V3/R1.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Applies fused add + RMSNorm to the input tensors.

        Args:
            hidden_states (torch.Tensor): Input tensor with shape [batch_size, hidden_size].
                                          Supports bfloat16.
            residual (torch.Tensor): Residual tensor with shape [batch_size, hidden_size].
                                     Supports bfloat16.
            weight (torch.Tensor): Weight tensor with shape [hidden_size].
                                   Supports bfloat16.
            eps (float, optional): Value added to denominator for numerical stability. Default: 1e-6.

        Returns:
            torch.Tensor: Output tensor with shape [batch_size, hidden_size], same dtype as hidden_states.
        """
        x = hidden_states.to(torch.float32) + residual.to(torch.float32)
        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        y = (x * inv_rms) * weight.to(torch.float32)
        return y.to(hidden_states.dtype)
