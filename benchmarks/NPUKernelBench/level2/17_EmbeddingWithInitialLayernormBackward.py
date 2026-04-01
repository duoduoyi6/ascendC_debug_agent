import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Embedding with Initial LayerNorm Backward Module.

    Backward pass for fused embedding + RMSNorm.
    Computes gradients w.r.t. embedding table and RMSNorm scale parameter.
    """
    def __init__(self):
        super(Model, self).__init__()
        self.vocab_size = 65536
        self.hidden_size = 4096

    def forward(self, grad_output: torch.Tensor, input_ids: torch.Tensor, hidden_states_fp32: torch.Tensor, rstd: torch.Tensor, norm_weight: torch.Tensor):
        """
        Backward pass for fused embedding + RMSNorm.

        Args:
            grad_output: (batch_size, seq_len, hidden_size) gradient from next layer
            input_ids: (batch_size, seq_len) token indices
            hidden_states_fp32: (batch_size, seq_len, hidden_size) saved hidden states
            rstd: (batch_size, seq_len, 1) reciprocal standard deviation
            norm_weight: (hidden_size,) RMSNorm scale parameter

        Returns:
            grad_embed_weight: (vocab_size, hidden_size) gradient for embedding table
            grad_norm_weight: (hidden_size,) gradient for RMSNorm weight
        """
        batch_size, seq_len, _ = grad_output.shape
        grad_output_f32 = grad_output.to(torch.float32)
        normalized = hidden_states_fp32 * rstd
        grad_norm_weight = (grad_output_f32 * normalized).sum(dim=(0, 1))
        grad_hidden = grad_output_f32 * norm_weight.to(torch.float32) * rstd
        grad_embed_weight = torch.zeros(self.vocab_size, self.hidden_size, dtype=torch.float32, device=grad_output.device)
        input_ids_flat = input_ids.reshape(-1)
        grad_hidden_flat = grad_hidden.reshape(-1, self.hidden_size)
        grad_embed_weight.index_add_(0, input_ids_flat, grad_hidden_flat)
        return grad_embed_weight.to(torch.bfloat16), grad_norm_weight.to(torch.bfloat16)
