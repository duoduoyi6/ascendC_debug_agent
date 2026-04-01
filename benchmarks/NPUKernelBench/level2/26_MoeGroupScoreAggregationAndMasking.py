import torch
import torch.nn as nn

class Model(nn.Module):
    """
    MoE Group-based Score Aggregation and Masking Module.

    Implements group-based routing for Mixture of Experts:
    1. Reshape expert scores into groups
    2. Compute top-2 scores per group and sum them for group quality
    3. Select top-k groups based on aggregated scores
    4. Mask out experts from non-selected groups
    """
    def __init__(self):
        super(Model, self).__init__()
        self.num_experts = 256
        self.n_group = 8
        self.topk_group = 4

    def forward(self, scores: torch.Tensor):
        """
        Group-based score aggregation and masking for MoE routing.

        Args:
            scores: Expert scores after sigmoid activation, shape (num_tokens, 256)

        Returns:
            masked_scores: Scores with non-selected groups masked, shape (num_tokens, 256)
            group_mask: Binary mask of selected groups, shape (num_tokens, 8)
        """
        experts_per_group = self.num_experts // self.n_group
        num_tokens = scores.size(0)
        group_scores_reshaped = scores.view(num_tokens, self.n_group, experts_per_group)
        top2_per_group = torch.topk(group_scores_reshaped, k=2, dim=-1)[0]
        group_scores = top2_per_group.sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(num_tokens, self.n_group, experts_per_group).reshape(num_tokens, self.num_experts)
        masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
        return masked_scores, group_mask
