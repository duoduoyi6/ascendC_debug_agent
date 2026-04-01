import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Backward pass for fused residual addition and RMSNorm.
    Computes gradients for hidden_states, residual, and weight given the gradient of the output.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        grad_output: torch.Tensor,
        x: torch.Tensor,
        normalized: torch.Tensor,
        rstd: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple:
        """
        Computes gradients for the fused residual + RMSNorm backward pass.

        Args:
            grad_output (torch.Tensor): Gradient from next layer with shape [batch_size, seq_len, hidden_size].
                                        Supports bfloat16.
            x (torch.Tensor): Saved tensor: hidden_states + residual from forward pass
                              with shape [batch_size, seq_len, hidden_size]. Supports float32.
            normalized (torch.Tensor): Saved tensor: normalized values from forward pass
                                       with shape [batch_size, seq_len, hidden_size]. Supports float32.
            rstd (torch.Tensor): Saved tensor: reciprocal standard deviation from forward pass
                                 with shape [batch_size, seq_len, 1]. Supports float32.
            weight (torch.Tensor): RMSNorm scale parameter with shape [hidden_size]. Supports float32.

        Returns:
            tuple: (grad_hidden_states, grad_residual, grad_weight)
                - grad_hidden_states (torch.Tensor): Gradient w.r.t. hidden_states
                                                     with shape [batch_size, seq_len, hidden_size].
                - grad_residual (torch.Tensor): Gradient w.r.t. residual
                                                with shape [batch_size, seq_len, hidden_size].
                - grad_weight (torch.Tensor): Gradient w.r.t. weight with shape [hidden_size].
        """
        grad_output_f32 = grad_output.to(torch.float32)

        grad_weight = (grad_output_f32 * normalized).sum(dim=(0, 1))

        grad_normalized = grad_output_f32 * weight

        mean_grad_norm = (grad_normalized * normalized).mean(dim=-1, keepdim=True)

        grad_x = rstd * (grad_normalized - mean_grad_norm * normalized)

        grad_x_bf16 = grad_x.to(torch.bfloat16)

        grad_hidden_states = grad_x_bf16
        grad_residual = grad_x_bf16.clone()

        return grad_hidden_states, grad_residual, grad_weight
