# torch.histc(input, bins=100, min=0, max=0, *, out=None) → Tensor
# https://docs.pytorch.org/docs/stable/generated/torch.histc.html#torch.histc

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that computes the histogram of a tensor.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, bins: int = 100, min_val: float = 0.0, max_val: float = 0.0) -> torch.Tensor:
        """
        Computes the histogram of the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            bins (int): Number of histogram bins.
            min_val (float): Lower end of the range (inclusive).
            max_val (float): Upper end of the range (inclusive). If 0, uses tensor max.

        Returns:
            torch.Tensor: Histogram tensor of shape (bins,).
        """
        return torch.histc(x, bins=bins, min=min_val, max=max_val)

INPUT_CASES = [{'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [128], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'bins', 'required': False, 'type': 'attr', 'value': 100},
             {'dtype': 'float', 'name': 'min_val', 'required': False, 'type': 'attr', 'value': 0.0},
             {'dtype': 'float',
              'name': 'max_val',
              'required': False,
              'type': 'attr',
              'value': 1.0}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [256], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'bins', 'required': False, 'type': 'attr', 'value': 50},
             {'dtype': 'float',
              'name': 'min_val',
              'required': False,
              'type': 'attr',
              'value': -1.0},
             {'dtype': 'float',
              'name': 'max_val',
              'required': False,
              'type': 'attr',
              'value': 1.0}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [512], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'bins', 'required': False, 'type': 'attr', 'value': 64},
             {'dtype': 'float', 'name': 'min_val', 'required': False, 'type': 'attr', 'value': 0.0},
             {'dtype': 'float',
              'name': 'max_val',
              'required': False,
              'type': 'attr',
              'value': 10.0}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [1024], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'bins', 'required': False, 'type': 'attr', 'value': 128},
             {'dtype': 'float', 'name': 'min_val', 'required': False, 'type': 'attr', 'value': 0.0},
             {'dtype': 'float',
              'name': 'max_val',
              'required': False,
              'type': 'attr',
              'value': 1.0}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [4096], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'bins', 'required': False, 'type': 'attr', 'value': 256},
             {'dtype': 'float',
              'name': 'min_val',
              'required': False,
              'type': 'attr',
              'value': -5.0},
             {'dtype': 'float',
              'name': 'max_val',
              'required': False,
              'type': 'attr',
              'value': 5.0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [16384],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'bins', 'required': False, 'type': 'attr', 'value': 100},
             {'dtype': 'float', 'name': 'min_val', 'required': False, 'type': 'attr', 'value': 0.0},
             {'dtype': 'float',
              'name': 'max_val',
              'required': False,
              'type': 'attr',
              'value': 1.0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [32768],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'bins', 'required': False, 'type': 'attr', 'value': 64},
             {'dtype': 'float', 'name': 'min_val', 'required': False, 'type': 'attr', 'value': 0.0},
             {'dtype': 'float',
              'name': 'max_val',
              'required': False,
              'type': 'attr',
              'value': 1.0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [65536],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'bins', 'required': False, 'type': 'attr', 'value': 128},
             {'dtype': 'float',
              'name': 'min_val',
              'required': False,
              'type': 'attr',
              'value': -1.0},
             {'dtype': 'float',
              'name': 'max_val',
              'required': False,
              'type': 'attr',
              'value': 1.0}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [100], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'bins', 'required': False, 'type': 'attr', 'value': 32},
             {'dtype': 'float', 'name': 'min_val', 'required': False, 'type': 'attr', 'value': 0.0},
             {'dtype': 'float',
              'name': 'max_val',
              'required': False,
              'type': 'attr',
              'value': 100.0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [18432],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'bins', 'required': False, 'type': 'attr', 'value': 100},
             {'dtype': 'float',
              'name': 'min_val',
              'required': False,
              'type': 'attr',
              'value': -2.0},
             {'dtype': 'float',
              'name': 'max_val',
              'required': False,
              'type': 'attr',
              'value': 2.0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [24576],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'bins', 'required': False, 'type': 'attr', 'value': 64},
             {'dtype': 'float', 'name': 'min_val', 'required': False, 'type': 'attr', 'value': 0.0},
             {'dtype': 'float',
              'name': 'max_val',
              'required': False,
              'type': 'attr',
              'value': 1.0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [128, 128],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'bins', 'required': False, 'type': 'attr', 'value': 100},
             {'dtype': 'float', 'name': 'min_val', 'required': False, 'type': 'attr', 'value': 0.0},
             {'dtype': 'float',
              'name': 'max_val',
              'required': False,
              'type': 'attr',
              'value': 1.0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [256, 256],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'bins', 'required': False, 'type': 'attr', 'value': 50},
             {'dtype': 'float',
              'name': 'min_val',
              'required': False,
              'type': 'attr',
              'value': -1.0},
             {'dtype': 'float',
              'name': 'max_val',
              'required': False,
              'type': 'attr',
              'value': 1.0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'bins', 'required': False, 'type': 'attr', 'value': 32},
             {'dtype': 'float', 'name': 'min_val', 'required': False, 'type': 'attr', 'value': 0.0},
             {'dtype': 'float',
              'name': 'max_val',
              'required': False,
              'type': 'attr',
              'value': 10.0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [128, 256],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'bins', 'required': False, 'type': 'attr', 'value': 128},
             {'dtype': 'float',
              'name': 'min_val',
              'required': False,
              'type': 'attr',
              'value': -3.0},
             {'dtype': 'float',
              'name': 'max_val',
              'required': False,
              'type': 'attr',
              'value': 3.0}]}]

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
