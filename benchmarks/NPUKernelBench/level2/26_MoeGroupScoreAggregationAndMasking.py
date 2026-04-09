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

INPUT_CASES = [{'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [1, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [2, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [4, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [8, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [16, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [32, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [64, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [128, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [256, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [512, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'scores',
              'required': True,
              'shape': [1, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'scores',
              'required': True,
              'shape': [2, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'scores',
              'required': True,
              'shape': [4, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'scores',
              'required': True,
              'shape': [8, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'scores',
              'required': True,
              'shape': [16, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'scores',
              'required': True,
              'shape': [32, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'scores',
              'required': True,
              'shape': [64, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'scores',
              'required': True,
              'shape': [128, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'scores',
              'required': True,
              'shape': [256, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'scores',
              'required': True,
              'shape': [512, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'scores',
              'required': True,
              'shape': [1, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'scores',
              'required': True,
              'shape': [2, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'scores',
              'required': True,
              'shape': [4, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'scores',
              'required': True,
              'shape': [8, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'scores',
              'required': True,
              'shape': [16, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'scores',
              'required': True,
              'shape': [32, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'scores',
              'required': True,
              'shape': [64, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'scores',
              'required': True,
              'shape': [128, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'scores',
              'required': True,
              'shape': [256, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'scores',
              'required': True,
              'shape': [512, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [3, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [5, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [7, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [11, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [13, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [17, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [19, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [23, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [29, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [31, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'scores',
              'required': True,
              'shape': [10, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'scores',
              'required': True,
              'shape': [20, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'scores',
              'required': True,
              'shape': [50, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'scores',
              'required': True,
              'shape': [100, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'scores',
              'required': True,
              'shape': [15, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'scores',
              'required': True,
              'shape': [25, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'scores',
              'required': True,
              'shape': [75, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'scores',
              'required': True,
              'shape': [150, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [200, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'scores',
              'required': True,
              'shape': [300, 256],
              'type': 'tensor'}]}]

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _make_boxes(shape, dtype):
    leading_shape = tuple(shape[:-1])
    mins = torch.rand(*leading_shape, 2, dtype=torch.float32)
    sizes = torch.rand(*leading_shape, 2, dtype=torch.float32) + 0.05
    maxs = mins + sizes
    boxes = torch.cat([mins, maxs], dim=-1)
    return boxes.to(dtype=dtype)


def _make_tensor(spec):
    dtype = _DTYPE_MAP[spec["dtype"]]
    shape = spec["shape"]
    name = spec["name"]
    value_range = spec.get("range")

    if dtype == torch.bool:
        return torch.randint(0, 2, tuple(shape), dtype=torch.int64).to(torch.bool)

    if name in {"boxes", "bboxes", "gtboxes"} and shape and shape[-1] == 4 and dtype in {
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    }:
        return _make_boxes(shape, dtype)

    if value_range is not None:
        low, high = value_range
        if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            high_exclusive = high + 1
            return torch.randint(low, high_exclusive, tuple(shape), dtype=dtype)
        return torch.empty(tuple(shape), dtype=dtype).uniform_(low, high)

    if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        return torch.randint(0, 17, tuple(shape), dtype=dtype)

    return torch.randn(*shape, dtype=dtype)


def _make_tensor_list(spec):
    dtype = _DTYPE_MAP[spec["dtype"]]
    return [torch.randn(*shape, dtype=dtype) for shape in spec["shapes"]]


def _make_arg(spec):
    spec_type = spec["type"]
    if spec_type == "tensor":
        return _make_tensor(spec)
    if spec_type == "tensor_list":
        return _make_tensor_list(spec)
    if spec_type == "attr":
        return spec["value"]
    raise ValueError(f"Unsupported input spec type: {spec_type}")


def get_input_groups():
    input_groups = []
    for case in INPUT_CASES:
        input_groups.append([_make_arg(spec) for spec in case["inputs"]])
    return input_groups


def get_init_inputs():
    return []
