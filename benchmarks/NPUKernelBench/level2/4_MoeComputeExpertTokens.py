import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that computes expert tokens for MoE (Mixture of Experts).
    torch_npu.npu_moe_compute_expert_tokens(sorted_expert_for_source_row, num_expert) -> Tensor

    Pure PyTorch implementation (replacing torch_npu.npu_moe_compute_expert_tokens):

        def forward(self, sorted_expert_for_source_row, num_expert):
            # sorted_expert_for_source_row: [N] int32 tensor with expert indices
            # num_expert: total number of experts

            # Count tokens per expert using bincount
            expert_tokens = torch.bincount(
                sorted_expert_for_source_row.long(),
                minlength=num_expert
            ).to(torch.int32)

            return expert_tokens
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, sorted_expert_for_source_row: torch.Tensor, num_expert: int) -> torch.Tensor:
        """
        Computes expert tokens for MoE routing.

        Args:
            sorted_expert_for_source_row (torch.Tensor): Result processed by experts, must be 1D.
                                                         dtype: int32, format: ND. Shape must be < 2147483647.
            num_expert (int): Total number of experts.

        Returns:
            torch.Tensor: Computed expert tokens tensor.
        """
        return torch_npu.npu_moe_compute_expert_tokens(sorted_expert_for_source_row, num_expert)

INPUT_CASES = [{'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [100],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [200],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [500],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [1000],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [100],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 16}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [200],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 16}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [500],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 16}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [1000],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 16}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [100],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 32}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [200],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 32}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [500],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 32}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [1000],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 32}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [100],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 64}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [200],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 64}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [500],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 64}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [1000],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 64}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [100],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 4}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [200],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 4}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [500],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 4}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [1000],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 4}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [2000],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [5000],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [10000],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [2000],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 16}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [5000],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 16}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [10000],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 16}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [2000],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 32}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [5000],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 32}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [10000],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 32}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [2000],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 64}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [5000],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 64}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [10000],
              'type': 'tensor'},
             {'dtype': 'int',
              'name': 'num_expert',
              'required': True,
              'type': 'attr',
              'value': 64}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [50],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [150],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [250],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [350],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [450],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [550],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [650],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [750],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [850],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [950],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [1500],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [2500],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [3500],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [4500],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [5500],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [6500],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [7500],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]},
 {'inputs': [{'dtype': 'int32',
              'name': 'sorted_expert_for_source_row',
              'required': True,
              'shape': [8500],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'num_expert', 'required': True, 'type': 'attr', 'value': 8}]}]

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
