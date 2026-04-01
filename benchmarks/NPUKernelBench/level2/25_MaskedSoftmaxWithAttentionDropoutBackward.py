import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Masked Softmax with Attention Dropout Backward Module.

    Backward pass for masked softmax with dropout.
    Computes gradients through three operations in reverse order:
    1. Dropout backward: scale by dropout mask and inverse probability
    2. Softmax backward: y * (grad - sum(y * grad))
    3. Masked fill backward: zero out gradients at masked positions
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        grad_output: torch.Tensor,
        p_attn: torch.Tensor,
        mask: torch.Tensor,
        dropout_mask: torch.Tensor,
        p_dropout: float,
    ) -> torch.Tensor:
        """
        Backward pass for masked softmax with dropout.

        Args:
            grad_output: Gradient w.r.t. output [B, H, T, T]
            p_attn: Softmax output (before dropout) [B, H, T, T]
            mask: Attention mask [B, 1, T, T] (True=unmasked, False=masked)
            dropout_mask: Dropout mask [B, H, T, T] (True=kept, False=dropped)
            p_dropout: Dropout probability

        Returns:
            grad_scores: Gradient w.r.t. input scores [B, H, T, T]
        """
        if p_dropout > 0.0:
            grad_softmax_output = grad_output * dropout_mask.float() / (1.0 - p_dropout)
        else:
            grad_softmax_output = grad_output

        sum_term = (p_attn * grad_softmax_output).sum(dim=-1, keepdim=True)
        grad_softmax_input = p_attn * (grad_softmax_output - sum_term)
        grad_scores = grad_softmax_input.masked_fill(~mask, 0.0)

        return grad_scores
