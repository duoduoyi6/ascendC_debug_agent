# torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None)
# https://docs.pytorch.org/docs/stable/generated/torch.topk.html#torch.topk

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that returns the k largest elements of a tensor.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, k: int, dim: int = -1, largest: bool = True) -> torch.Tensor:
        """
        Returns the k largest/smallest elements of the input tensor along a given dimension.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            k (int): The number of elements to return.
            dim (int): The dimension to sort along.
            largest (bool): If True, return largest k elements; otherwise smallest.

        Returns:
            torch.Tensor: Tensor of top k values with shape [..., k, ...].
        """
        values, indices = torch.topk(x, k, dim=dim, largest=largest)
        return values

INPUT_CASES = [{'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [128], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 10},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [256], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 20},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': False}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [512], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 50},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [1024], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 100},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [128, 128],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 16},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [128, 128],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 32},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': False}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [256, 256],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 64},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [256, 256],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 16},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -2},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': False}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 8},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [128, 256],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 32},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [64, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 16},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [64, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 8},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': False}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [64, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 32},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 2},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [64, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 16},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': False}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [32, 32, 32],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 8},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -2},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [16, 16, 16, 16],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 4},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [16, 16, 16, 16],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 8},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': False}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 16, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 16},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 3, 224, 224],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 32},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [1, 64, 56, 56],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 8},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 2},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': False}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [4096, 18432],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 64},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [8192, 16384],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 128},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [100], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 10},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': False}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [100, 2007],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 50},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [17, 301],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 15},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [7, 15, 23],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 5},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [7, 15, 23],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 8},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': False}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [3, 5, 7, 11],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 3},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': True}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [1, 5, 111, 111],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'k', 'required': True, 'type': 'attr', 'value': 16},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 2},
             {'dtype': 'bool',
              'name': 'largest',
              'required': False,
              'type': 'attr',
              'value': False}]}]

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
