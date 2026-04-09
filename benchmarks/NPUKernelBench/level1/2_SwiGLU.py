import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a SwiGLU activation.
    SwiGLU(x, dim) = Swish(a) * b, where a and b are chunks of x along dim.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Applies SwiGLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor where the size of dim must be even.
            dim (int, optional): The dimension along which to chunk the tensor.

        Returns:
            torch.Tensor: Output tensor with SwiGLU applied, shape is same as x except
                          dim is halved.
        """
        a, b = torch.chunk(x, 2, dim=dim)
        return torch.nn.functional.silu(a) * b

INPUT_CASES = [{'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [130], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [256], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [512], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [1024], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [2048], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float32', 'name': 'x', 'required': True, 'shape': [4096], 'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [128, 128],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [128, 128],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [256, 256],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [256, 256],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [512, 512],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [512, 512],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [1024, 1024],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [1024, 1024],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [128, 256],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [128, 256],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [256, 512],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [256, 512],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [64, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [64, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [64, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [32, 32, 32],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [32, 32, 32],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [32, 32, 32],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [128, 64, 32],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [128, 64, 32],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [128, 64, 32],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [16, 16, 16, 16],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [16, 16, 16, 16],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [16, 16, 16, 16],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 16, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 16, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 64, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 64, 64, 64],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [1, 128, 32, 32],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [1, 128, 32, 32],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [1, 256, 16, 16],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [1, 256, 16, 16],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [4096, 8192],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [349, 1536],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [5007, 3840],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [1829, 3072],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [109, 5120],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [147, 10240],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [221, 12288],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [2677, 13824],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [1001, 18432],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [1776, 24576],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]}]

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
