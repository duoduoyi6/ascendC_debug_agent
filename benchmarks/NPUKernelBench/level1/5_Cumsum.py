import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs cumulative sum along a specified dimension.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Applies cumulative sum along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            dim (int): The dimension to do the operation over.

        Returns:
            torch.Tensor: Output tensor with cumulative sum, same shape as input.
        """
        return torch.cumsum(x, dim=dim)

INPUT_CASES = [{'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [128], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [256], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [512], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [1024], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [2048], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [128, 128],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [128, 128],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [256, 256],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [256, 256],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -2}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [128, 256],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [128, 256],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [256, 512],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [256, 512],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -2}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [64, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [64, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [64, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 2}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [64, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [32, 32, 32],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -2}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [32, 32, 32],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -3}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [128, 64, 32],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [128, 64, 32],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [16, 16, 16, 16],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [16, 16, 16, 16],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [16, 16, 16, 16],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 2}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [16, 16, 16, 16],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 3}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [16, 16, 16, 16],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [1, 16, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [1, 16, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 2}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [1, 16, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 3, 224, 224],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 2}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 3, 224, 224],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 3}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [1, 64, 56, 56],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -2}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [1, 128, 28, 28],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -3}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [4096, 18432],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [4096, 18432],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [8192, 16384],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [8192, 16384],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -2}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [100], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [100, 2007],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [100, 2007],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [17, 301],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [13, 2117],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -2}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [7, 15, 23],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [7, 15, 23],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [7, 15, 23],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 2}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [3, 5, 7, 11],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [3, 5, 7, 11],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [1, 5, 111, 111],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 2}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [1, 5, 111, 111],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': True, 'type': 'attr', 'value': 3}]}]

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
