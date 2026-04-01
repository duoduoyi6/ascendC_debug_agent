import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Attention Softmax with Softcapping and Dropout Module.

    Applies Gemma3's softcapping transformation followed by softmax normalization.
    Softcapping: tanh(logits / 30.0) * 30.0
    This clamps effective logit range to approximately [-30, +30].
    """
    def __init__(self):
        super(Model, self).__init__()
        self.SOFTCAP = 30.0

    def forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Apply softcapping transformation followed by softmax.

        Args:
            attn_weights: Attention logits of shape (batch_size, num_heads, seq_len_q, seq_len_k)

        Returns:
            Normalized attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        scaled = attn_weights / self.SOFTCAP
        clamped = torch.tanh(scaled)
        softcapped = clamped * self.SOFTCAP
        output = F.softmax(softcapped, dim=-1, dtype=torch.float32).to(attn_weights.dtype)
        return output
