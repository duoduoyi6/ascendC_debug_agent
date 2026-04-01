import torch
import torch.nn as nn

class Model(nn.Module):
    """
    KV Cache Update with RoPE Backward Module.

    Backward pass for KV cache update with Rotary Position Encoding.
    Computes gradients for key states, value states, cos, sin, and cache tensors.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_key_cache: torch.Tensor, grad_value_cache: torch.Tensor, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache_position: torch.Tensor):
        """
        Backward pass for KV cache update with RoPE.

        Args:
            grad_key_cache: Gradient of key cache [batch_size, num_kv_heads, max_seq_len, head_dim]
            grad_value_cache: Gradient of value cache [batch_size, num_kv_heads, max_seq_len, head_dim]
            key_states: Key states [batch_size, num_kv_heads, new_seq_len, head_dim]
            cos: Cosine values for RoPE [batch_size, new_seq_len, head_dim]
            sin: Sine values for RoPE [batch_size, new_seq_len, head_dim]
            cache_position: Position indices in cache [new_seq_len]

        Returns:
            grad_key_states: Gradient for key states
            grad_value_states: Gradient for value states
            grad_cos: Gradient for cos
            grad_sin: Gradient for sin
            grad_key_cache_input: Gradient for key cache input
            grad_value_cache_input: Gradient for value cache input
        """
        half_dim = key_states.shape[-1] // 2
        k1 = key_states[..., :half_dim]
        k2 = key_states[..., half_dim:]
        k_rotated_half = torch.cat((-k2, k1), dim=-1)
        cos_expanded = cos.unsqueeze(1)
        sin_expanded = sin.unsqueeze(1)
        grad_key_states_rotated = grad_key_cache[:, :, cache_position]
        grad_value_states = grad_value_cache[:, :, cache_position]
        grad_from_cos_term = grad_key_states_rotated * cos_expanded
        grad_k_rotated_half = grad_key_states_rotated * sin_expanded

        grad_k_rotated_half_1 = grad_k_rotated_half[..., :half_dim]
        grad_k_rotated_half_2 = grad_k_rotated_half[..., half_dim:]

        grad_k2_from_rotate = -grad_k_rotated_half_1
        grad_k1_from_rotate = grad_k_rotated_half_2

        grad_k1_total = grad_from_cos_term[..., :half_dim] + grad_k1_from_rotate
        grad_k2_total = grad_from_cos_term[..., half_dim:] + grad_k2_from_rotate

        grad_key_states = torch.cat([grad_k1_total, grad_k2_total], dim=-1)

        grad_cos_expanded = grad_key_states_rotated * key_states
        grad_cos = grad_cos_expanded.sum(dim=1)

        grad_sin_expanded = grad_key_states_rotated * k_rotated_half
        grad_sin = grad_sin_expanded.sum(dim=1)

        grad_key_cache_input = grad_key_cache.clone()
        grad_key_cache_input[:, :, cache_position] = 0

        grad_value_cache_input = grad_value_cache.clone()
        grad_value_cache_input[:, :, cache_position] = 0

        return (
            grad_key_states.to(torch.bfloat16),
            grad_value_states.to(torch.bfloat16),
            grad_cos.to(torch.bfloat16),
            grad_sin.to(torch.bfloat16),
            grad_key_cache_input.to(torch.bfloat16),
            grad_value_cache_input.to(torch.bfloat16),
        )
