import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs interleave RoPE (Rotary Position Embedding).
    torch_npu.npu_interleave_rope(x, cos, sin) -> Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    # PyTorch native implementation of forward function
    # def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    #     orig_dtype = x.dtype

    #     x_f32 = x.float()
    #     cos_f32 = cos.float()
    #     sin_f32 = sin.float()

    #     B, N, S, D = x_f32.shape

    #     x_reshaped = x_f32.reshape(B, N, S, D // 2, 2)
    #     x_transposed = x_reshaped.transpose(-1, -2)
    #     x_interleaved = x_transposed.reshape(B, N, S, D)

    #     cos_expanded = cos_f32
    #     sin_expanded = sin_f32

    #     if cos_expanded.shape[2] == 1 and S > 1:
    #         cos_expanded = cos_expanded.expand(B, N, S, D)
    #     if sin_expanded.shape[2] == 1 and S > 1:
    #         sin_expanded = sin_expanded.expand(B, N, S, D)

    #     x_rotated = torch.zeros_like(x_interleaved)
    #     x_rotated[..., :D // 2] = -x_interleaved[..., D // 2:]
    #     x_rotated[..., D // 2:] = x_interleaved[..., :D // 2]

    #     output_f32 = x_interleaved * cos_expanded + x_rotated * sin_expanded

    #     return output_f32.to(orig_dtype)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Performs interleave RoPE on input tensor.

        Args:
            x (torch.Tensor): Input tensor to process. Must be 4D with shape (B, N, S, D).
                              dtype: bfloat16, float16, format: ND.
                              Does not support non-contiguous tensors.
            cos (torch.Tensor): RoPE cosine component. Must be 4D with shape (B, N, S, D).
                                S can be 1 or same as x's S. dtype and format must match x.
                                Does not support non-contiguous tensors.
            sin (torch.Tensor): RoPE sine component. Shape, dtype and format must match cos.
                                Does not support non-contiguous tensors.

        Returns:
            torch.Tensor: Output tensor after interleave RoPE, same shape as input x.
        """
        return torch_npu.npu_interleave_rope(x, cos, sin)
