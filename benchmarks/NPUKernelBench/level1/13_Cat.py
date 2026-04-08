# torch.cat(tensors, dim=0, *, out=None) → Tensor
# https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that concatenates tensors along a dimension.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tensors: list, dim: int = 0) -> torch.Tensor:
        """
        Concatenates the given sequence of tensors in the given dimension.

        Args:
            tensors (list): List of tensors to concatenate. All tensors must have the same shape except in the concatenating dimension.
            dim (int, optional): The dimension over which the tensors are concatenated.

        Returns:
            torch.Tensor: Concatenated tensor.
        """
        return torch.cat(tensors, dim=dim)

INPUT_CASES = [{'inputs': [{'dim': 0,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[128], [128], [128]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[256], [256]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[512], [512], [512], [512]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[64, 64], [64, 64]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[64, 64], [64, 64], [64, 64]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[128, 128], [128, 128]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'bfloat16',
              'name': 'tensors',
              'required': True,
              'shapes': [[256, 256], [256, 256]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 1,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[64, 128], [64, 256], [64, 64]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dim': 1,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[128, 64], [128, 128], [128, 32]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dim': 1,
              'dtype': 'bfloat16',
              'name': 'tensors',
              'required': True,
              'shapes': [[256, 128], [256, 256]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[64, 64, 64], [64, 64, 64]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[32, 64, 64], [32, 64, 64], [32, 64, 64]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 1,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[64, 32, 64], [64, 32, 64]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dim': 2,
              'dtype': 'bfloat16',
              'name': 'tensors',
              'required': True,
              'shapes': [[64, 64, 32], [64, 64, 32], [64, 64, 32]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 2}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[1, 64, 56, 56], [1, 64, 56, 56]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[2, 64, 56, 56], [2, 64, 56, 56], [2, 64, 56, 56]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 1,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[1, 32, 56, 56], [1, 64, 56, 56], [1, 128, 56, 56]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dim': 2,
              'dtype': 'bfloat16',
              'name': 'tensors',
              'required': True,
              'shapes': [[1, 64, 28, 56], [1, 64, 28, 56]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 2}]},
 {'inputs': [{'dim': 3,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[1, 64, 56, 28], [1, 64, 56, 28], [1, 64, 56, 28]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 3}]},
 {'inputs': [{'dim': -1,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[1, 64, 56, 56], [1, 64, 56, 56]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': -1}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[4096, 4096], [4096, 4096]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 1,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[4096, 4096], [4096, 4096]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[2048, 8192], [2048, 8192], [2048, 8192]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 1,
              'dtype': 'bfloat16',
              'name': 'tensors',
              'required': True,
              'shapes': [[8192, 2048], [8192, 4096]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[64, 128, 256], [64, 128, 256]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 1,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[64, 64, 256], [64, 128, 256], [64, 64, 256]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dim': 0,
              'dtype': 'bfloat16',
              'name': 'tensors',
              'required': True,
              'shapes': [[32, 64, 128, 256], [32, 64, 128, 256]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[100, 200], [100, 200]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[17, 31], [17, 31], [17, 31]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 1,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[100, 100], [100, 200], [100, 50]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[7, 15, 23], [7, 15, 23]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'bfloat16',
              'name': 'tensors',
              'required': True,
              'shapes': [[11, 19, 29], [11, 19, 29], [11, 19, 29]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[3, 5, 7, 11], [3, 5, 7, 11]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[1, 7, 13, 17], [1, 7, 13, 17], [1, 7, 13, 17]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'bfloat16',
              'name': 'tensors',
              'required': True,
              'shapes': [[1, 5, 111, 111], [1, 5, 111, 111]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[2, 33, 55, 77], [2, 33, 55, 77]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[1, 13, 65, 65], [1, 13, 65, 65]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[1, 11, 113, 113], [1, 11, 113, 113]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[123, 6144], [123, 6144]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'bfloat16',
              'name': 'tensors',
              'required': True,
              'shapes': [[789, 12288], [789, 12288]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[4096, 11008], [4096, 11008]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[4096, 13824], [4096, 13824], [4096, 13824]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'bfloat16',
              'name': 'tensors',
              'required': True,
              'shapes': [[4096, 24576], [4096, 24576]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[8192, 16384], [8192, 16384]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[2048, 13824], [2048, 13824]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 1,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[128, 256], [128, 512], [128, 128]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dim': 1,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[64, 128], [64, 256], [64, 64], [64, 32]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dim': 0,
              'dtype': 'bfloat16',
              'name': 'tensors',
              'required': True,
              'shapes': [[32, 64], [32, 64], [32, 64], [32, 64]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[16, 16], [16, 16], [16, 16], [16, 16], [16, 16]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float16',
              'name': 'tensors',
              'required': True,
              'shapes': [[1, 3584], [1, 3584]],
              'type': 'tensor_list'},
             {'dtype': 'int', 'name': 'dim', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dim': 0,
              'dtype': 'float32',
              'name': 'tensors',
              'required': True,
              'shapes': [[1, 5120], [1, 5120], [1, 5120]],
              'type': 'tensor_list'},
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
