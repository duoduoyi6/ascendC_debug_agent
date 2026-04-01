import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Adaptive Instance Normalization 2D Backward Module.

    This module computes the backward pass (gradients) for Adaptive Instance Normalization 2D.
    It calculates gradients with respect to the input tensor, weight (gamma), and bias (beta)
    based on the gradient of the loss with respect to the output.

    The forward pass of AdaIN performs: output = weight * (x - mean) / std + bias
    This module computes the gradients needed for backpropagation.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        """
        Backward pass for Adaptive Instance Normalization 2D.

        Computes gradients with respect to input, weight (gamma), and bias (beta).

        Args:
            grad_output: Gradient of loss w.r.t. output, shape (N, C, H, W)
            x: Original input tensor from forward pass, shape (N, C, H, W)
            weight: Scale parameter (gamma), shape (C,)
            mean: Mean computed in forward pass, shape (N, C, 1, 1)
            std: Standard deviation computed in forward pass, shape (N, C, 1, 1)

        Returns:
            grad_input: Gradient w.r.t. input, shape (N, C, H, W)
            grad_weight: Gradient w.r.t. weight (gamma), shape (C,)
            grad_bias: Gradient w.r.t. bias (beta), shape (C,)
        """
        N, C, H, W = x.shape
        spatial_size = H * W
        x_centered = x - mean
        x_normalized = x_centered / std
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        grad_weight = (grad_output * x_normalized).sum(dim=(0, 2, 3))
        weight_reshaped = weight.view(1, C, 1, 1)
        grad_output_scaled = grad_output * weight_reshaped
        grad_var = (grad_output_scaled * x_centered).sum(dim=(2, 3), keepdim=True) * (-0.5) * torch.pow(std, -3)
        grad_mean = grad_output_scaled.sum(dim=(2, 3), keepdim=True) * (-1.0 / std) + grad_var * (-2.0 * x_centered.mean(dim=(2, 3), keepdim=True))
        grad_input = grad_output_scaled / std + grad_var * 2.0 * x_centered / spatial_size + grad_mean / spatial_size
        return grad_input, grad_weight, grad_bias
