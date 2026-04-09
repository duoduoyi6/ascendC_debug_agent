import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs top-k and top-p filtering.
    torch_npu.npu_top_k_top_p(logits, p, k) -> torch.Tensor
    """
    def __init__(self):
        super(Model, self).__init__()

    # PyTorch native implementation of forward function
    # def forward(self, logits: torch.Tensor, p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    #     ori_dtype = logits.dtype
    #     p = p.to(torch.float32)

    #     logits_sort, logits_idx = logits.sort(dim=-1, descending=False, stable=True)
    #     kth_idx = logits_sort.size(1) - k.to(torch.long)
    #     kth_value = logits_sort.gather(1, kth_idx.unsqueeze(dim=1))
    #     top_k_mask = logits_sort < kth_value
    #     logits_sort.masked_fill_(top_k_mask, -float("inf"))

    #     softmax_res = logits_sort.to(torch.float32).softmax(dim=-1)
    #     cumsum_res = softmax_res.cumsum(dim=-1)
    #     top_p_mask = cumsum_res <= 1 - p.unsqueeze(dim=1)
    #     top_p_mask[:, -1] = False
    #     logits_sort.masked_fill_(top_p_mask, -float("inf"))

    #     logits = torch.empty_like(logits_sort).scatter_(dim=-1, index=logits_idx, src=logits_sort)

    #     return logits.to(ori_dtype)

    def forward(self, logits: torch.Tensor, p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Performs top-k and top-p filtering on logits.

        Args:
            logits (torch.Tensor): Data to process. Must be 2D.
                                   dtype: float32, float16, bfloat16, format: ND.
                                   Supports non-contiguous tensors.
            p (torch.Tensor): Top-p threshold tensor. Range: [0, 1].
                              dtype: float32, float16, bfloat16 (must match logits).
                              Must be 1D with size matching logits' first dimension.
                              format: ND, supports non-contiguous tensors.
            k (torch.Tensor): Top-k threshold tensor. Range: [1, 1024], max <= logits.size(1).
                              dtype: int32. Must be 1D with size matching logits' first dimension.
                              format: ND, supports non-contiguous tensors.

        Returns:
            torch.Tensor: Filtered logits tensor.
        """
        return torch_npu.npu_top_k_top_p(logits, p, k)
