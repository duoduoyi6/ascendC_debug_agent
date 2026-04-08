# torch_npu.npu_dynamic_quant(x, *, smooth_scales=None, group_index=None, dst_type=None) ->(Tensor, Tensor)
# https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_dynamic_quant.md

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs dynamic quantization on NPU.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, smooth_scales: torch.Tensor = None,
                group_index: torch.Tensor = None, dst_type=None):
        """
        Performs dynamic quantization on the input tensor.

        Args:
            x (torch.Tensor): Input tensor to be quantized.
            smooth_scales (torch.Tensor, optional): Smooth scale factors.
            group_index (torch.Tensor, optional): Group indices for per-group quantization.
            dst_type (optional): Target data type for quantized output.

        Returns:
            tuple: (quantized_tensor, scale_tensor)
        """
        import torch_npu
        return torch_npu.npu_dynamic_quant(x, smooth_scales=smooth_scales,
                                            group_index=group_index, dst_type=dst_type)

INPUT_CASES = [{'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [128, 128],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [256, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [512, 512],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [1024, 1024],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [128, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [256, 512],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [512, 1024],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [1024, 2048],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [64, 64, 64],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [32, 32, 32],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [128, 64, 32],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [64, 32, 16],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [4096, 11008],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [4096, 14336],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [4096, 12288],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [4096, 13824],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [4096, 18432],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [4096, 24576],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [8192, 16384],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [2048, 13824],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [5120, 27648],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [3584, 18944],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [5120, 13824],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [1536, 8960],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [2560, 14592],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [3072, 12288],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [6144, 20480],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [100, 200],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [17, 301],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [13, 211],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [7, 15, 23],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [11, 19, 29],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [123, 6144],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [789, 12288],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [1, 4096],
              'type': 'tensor'},
             {'dtype': 'float16',
              'name': 'smooth_scales',
              'required': False,
              'shape': [4096],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [128, 4096],
              'type': 'tensor'},
             {'dtype': 'float16',
              'name': 'smooth_scales',
              'required': False,
              'shape': [4096],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [256, 8192],
              'type': 'tensor'},
             {'dtype': 'bfloat16',
              'name': 'smooth_scales',
              'required': False,
              'shape': [8192],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [4096, 11008],
              'type': 'tensor'},
             {'dtype': 'float16',
              'name': 'smooth_scales',
              'required': False,
              'shape': [11008],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [2048, 5120],
              'type': 'tensor'},
             {'dtype': 'bfloat16',
              'name': 'smooth_scales',
              'required': False,
              'shape': [5120],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [128, 4096],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [256, 8192],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [512, 4096],
              'type': 'tensor'},
             {'dtype': 'bfloat16',
              'name': 'smooth_scales',
              'required': False,
              'shape': [4096],
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
