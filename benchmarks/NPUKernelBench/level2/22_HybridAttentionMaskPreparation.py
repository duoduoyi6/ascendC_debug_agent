import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Hybrid causal mask preparation system that creates separate attention masks
    for full attention and sliding window attention layers.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        batch_size_scalar: int,
        seq_length_scalar: int,
        past_key_values_length_scalar: int,
        num_attention_heads: int = 64,
        swa_num_attention_heads: int = 64,
        sliding_window: int = 128,
    ) -> tuple:
        """
        Creates hybrid attention masks for full and sliding window attention.

        Args:
            batch_size_scalar (int): Batch size.
            seq_length_scalar (int): Current sequence length.
            past_key_values_length_scalar (int): Length of cached keys/values.
            num_attention_heads (int, optional): Number of heads for full attention. Default: 64.
            swa_num_attention_heads (int, optional): Number of heads for sliding window attention. Default: 64.
            sliding_window (int, optional): Window size for sliding window attention. Default: 128.

        Returns:
            tuple: (full_attention_mask, sliding_window_attention_mask)
                - full_attention_mask (torch.Tensor): Full causal attention mask
                                                      with shape [batch_size, num_attention_heads, seq_length, total_length].
                - sliding_window_attention_mask (torch.Tensor): Sliding window causal attention mask
                                                                with shape [batch_size, swa_num_attention_heads, seq_length, total_length].
        """
        batch_size = int(batch_size_scalar)
        seq_length = int(seq_length_scalar)
        past_key_values_length = int(past_key_values_length_scalar)

        target_length = seq_length
        source_length = seq_length + past_key_values_length

        full_mask = torch.ones(
            (target_length, source_length),
            dtype=torch.bool
        )

        target_indices = torch.arange(target_length)[:, None]
        source_indices = torch.arange(source_length)[None, :]
        causal_cond = target_indices >= (source_indices - past_key_values_length)
        full_mask = full_mask.masked_fill(causal_cond, False)

        full_attention_mask = full_mask[None, None, :, :].expand(
            batch_size, num_attention_heads, target_length, source_length
        ).contiguous()

        swa_mask = torch.zeros(
            (target_length, source_length),
            dtype=torch.bool
        )

        window_cond = (source_indices - past_key_values_length) >= (target_indices - sliding_window)

        valid_positions = causal_cond & window_cond
        swa_mask = swa_mask.masked_fill(valid_positions, False)

        sliding_window_attention_mask = swa_mask[None, None, :, :].expand(
            batch_size, swa_num_attention_heads, target_length, source_length
        ).contiguous()

        return full_attention_mask, sliding_window_attention_mask
