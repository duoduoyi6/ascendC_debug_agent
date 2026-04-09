# torch_npu.npu_iou(bboxes, gtboxes, mode=0) -> Tensor
# https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/%EF%BC%88beta%EF%BC%89torch_npu-npu_iou.md

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that computes IoU (Intersection over Union) on NPU.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, bboxes: torch.Tensor, gtboxes: torch.Tensor, mode: int = 0) -> torch.Tensor:
        """
        Computes IoU between bounding boxes.

        Args:
            bboxes (torch.Tensor): First set of bounding boxes.
            gtboxes (torch.Tensor): Second set of bounding boxes (ground truth).
            mode (int, optional): IoU computation mode (0: IoU, 1: IoF).

        Returns:
            torch.Tensor: IoU values between bounding boxes.
        """
        import torch_npu
        return torch_npu.npu_iou(bboxes, gtboxes, mode=mode)

INPUT_CASES = [{'inputs': [{'dtype': 'float32',
              'name': 'bboxes',
              'required': True,
              'shape': [100, 4],
              'type': 'tensor'},
             {'dtype': 'float32',
              'name': 'gtboxes',
              'required': True,
              'shape': [10, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'bboxes',
              'required': True,
              'shape': [100, 4],
              'type': 'tensor'},
             {'dtype': 'float32',
              'name': 'gtboxes',
              'required': True,
              'shape': [10, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'bboxes',
              'required': True,
              'shape': [100, 4],
              'type': 'tensor'},
             {'dtype': 'float32',
              'name': 'gtboxes',
              'required': True,
              'shape': [50, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'bboxes',
              'required': True,
              'shape': [100, 4],
              'type': 'tensor'},
             {'dtype': 'float32',
              'name': 'gtboxes',
              'required': True,
              'shape': [100, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'bboxes',
              'required': True,
              'shape': [256, 4],
              'type': 'tensor'},
             {'dtype': 'float16',
              'name': 'gtboxes',
              'required': True,
              'shape': [10, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'bboxes',
              'required': True,
              'shape': [256, 4],
              'type': 'tensor'},
             {'dtype': 'float16',
              'name': 'gtboxes',
              'required': True,
              'shape': [50, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'bboxes',
              'required': True,
              'shape': [256, 4],
              'type': 'tensor'},
             {'dtype': 'float16',
              'name': 'gtboxes',
              'required': True,
              'shape': [100, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'bboxes',
              'required': True,
              'shape': [512, 4],
              'type': 'tensor'},
             {'dtype': 'float32',
              'name': 'gtboxes',
              'required': True,
              'shape': [10, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'bboxes',
              'required': True,
              'shape': [512, 4],
              'type': 'tensor'},
             {'dtype': 'float32',
              'name': 'gtboxes',
              'required': True,
              'shape': [50, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'bboxes',
              'required': True,
              'shape': [512, 4],
              'type': 'tensor'},
             {'dtype': 'float32',
              'name': 'gtboxes',
              'required': True,
              'shape': [100, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'bboxes',
              'required': True,
              'shape': [1024, 4],
              'type': 'tensor'},
             {'dtype': 'float16',
              'name': 'gtboxes',
              'required': True,
              'shape': [10, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'bboxes',
              'required': True,
              'shape': [1024, 4],
              'type': 'tensor'},
             {'dtype': 'float16',
              'name': 'gtboxes',
              'required': True,
              'shape': [50, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'bboxes',
              'required': True,
              'shape': [1024, 4],
              'type': 'tensor'},
             {'dtype': 'float16',
              'name': 'gtboxes',
              'required': True,
              'shape': [100, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'bboxes',
              'required': True,
              'shape': [2048, 4],
              'type': 'tensor'},
             {'dtype': 'float32',
              'name': 'gtboxes',
              'required': True,
              'shape': [10, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'bboxes',
              'required': True,
              'shape': [2048, 4],
              'type': 'tensor'},
             {'dtype': 'float32',
              'name': 'gtboxes',
              'required': True,
              'shape': [50, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'bboxes',
              'required': True,
              'shape': [2048, 4],
              'type': 'tensor'},
             {'dtype': 'float32',
              'name': 'gtboxes',
              'required': True,
              'shape': [100, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'bboxes',
              'required': True,
              'shape': [4096, 4],
              'type': 'tensor'},
             {'dtype': 'float16',
              'name': 'gtboxes',
              'required': True,
              'shape': [10, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'bboxes',
              'required': True,
              'shape': [4096, 4],
              'type': 'tensor'},
             {'dtype': 'float16',
              'name': 'gtboxes',
              'required': True,
              'shape': [50, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'bboxes',
              'required': True,
              'shape': [4096, 4],
              'type': 'tensor'},
             {'dtype': 'float16',
              'name': 'gtboxes',
              'required': True,
              'shape': [100, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'bboxes',
              'required': True,
              'shape': [100, 4],
              'type': 'tensor'},
             {'dtype': 'float32',
              'name': 'gtboxes',
              'required': True,
              'shape': [1, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'bboxes',
              'required': True,
              'shape': [1000, 4],
              'type': 'tensor'},
             {'dtype': 'float32',
              'name': 'gtboxes',
              'required': True,
              'shape': [1, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'bboxes',
              'required': True,
              'shape': [100, 4],
              'type': 'tensor'},
             {'dtype': 'float16',
              'name': 'gtboxes',
              'required': True,
              'shape': [200, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'bboxes',
              'required': True,
              'shape': [500, 4],
              'type': 'tensor'},
             {'dtype': 'float16',
              'name': 'gtboxes',
              'required': True,
              'shape': [150, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'bboxes',
              'required': True,
              'shape': [1500, 4],
              'type': 'tensor'},
             {'dtype': 'float32',
              'name': 'gtboxes',
              'required': True,
              'shape': [75, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'bboxes',
              'required': True,
              'shape': [3000, 4],
              'type': 'tensor'},
             {'dtype': 'float16',
              'name': 'gtboxes',
              'required': True,
              'shape': [25, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'bboxes',
              'required': True,
              'shape': [8192, 4],
              'type': 'tensor'},
             {'dtype': 'float32',
              'name': 'gtboxes',
              'required': True,
              'shape': [10, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'bboxes',
              'required': True,
              'shape': [8192, 4],
              'type': 'tensor'},
             {'dtype': 'float16',
              'name': 'gtboxes',
              'required': True,
              'shape': [50, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 1}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'bboxes',
              'required': True,
              'shape': [8192, 4],
              'type': 'tensor'},
             {'dtype': 'float32',
              'name': 'gtboxes',
              'required': True,
              'shape': [100, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'bboxes',
              'required': True,
              'shape': [64, 4],
              'type': 'tensor'},
             {'dtype': 'float16',
              'name': 'gtboxes',
              'required': True,
              'shape': [8, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'bboxes',
              'required': True,
              'shape': [128, 4],
              'type': 'tensor'},
             {'dtype': 'float32',
              'name': 'gtboxes',
              'required': True,
              'shape': [16, 4],
              'type': 'tensor'},
             {'dtype': 'int', 'name': 'mode', 'required': False, 'type': 'attr', 'value': 1}]}]

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
