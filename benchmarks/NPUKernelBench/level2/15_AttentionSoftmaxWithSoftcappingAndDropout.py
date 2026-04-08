import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Attention Softmax with Softcapping and Dropout Module.

    Applies Gemma3's softcapping transformation followed by softmax normalization.
    Softcapping: tanh(logits / 30.0) * 30.0
    This clamps effective logit range to approximately [-30, +30].
    """
    def __init__(self):
        super(Model, self).__init__()
        self.SOFTCAP = 30.0

    def forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Apply softcapping transformation followed by softmax.

        Args:
            attn_weights: Attention logits of shape (batch_size, num_heads, seq_len_q, seq_len_k)

        Returns:
            Normalized attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        scaled = attn_weights / self.SOFTCAP
        clamped = torch.tanh(scaled)
        softcapped = clamped * self.SOFTCAP
        output = F.softmax(softcapped, dim=-1, dtype=torch.float32).to(attn_weights.dtype)
        return output

INPUT_CASES = [{'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 1, 16, 16],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 2, 32, 32],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 4, 64, 64],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 8, 128, 128],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 12, 256, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'attn_weights',
              'required': True,
              'shape': [2, 1, 16, 16],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'attn_weights',
              'required': True,
              'shape': [2, 2, 32, 32],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'attn_weights',
              'required': True,
              'shape': [2, 4, 64, 64],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'attn_weights',
              'required': True,
              'shape': [2, 8, 128, 128],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'attn_weights',
              'required': True,
              'shape': [2, 12, 256, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'attn_weights',
              'required': True,
              'shape': [4, 1, 16, 16],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'attn_weights',
              'required': True,
              'shape': [4, 2, 32, 32],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'attn_weights',
              'required': True,
              'shape': [4, 4, 64, 64],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'attn_weights',
              'required': True,
              'shape': [4, 8, 128, 128],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'attn_weights',
              'required': True,
              'shape': [4, 12, 256, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 16, 512, 512],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 32, 1024, 1024],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 8, 64, 64],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 8, 128, 128],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 8, 256, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 1, 128, 128],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 4, 256, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 16, 512, 512],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'attn_weights',
              'required': True,
              'shape': [2, 8, 512, 512],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'attn_weights',
              'required': True,
              'shape': [2, 16, 1024, 1024],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 1, 64, 128],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 2, 128, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 4, 256, 512],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'attn_weights',
              'required': True,
              'shape': [2, 8, 512, 1024],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 12, 384, 384],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 3, 224, 224],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 6, 112, 112],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'attn_weights',
              'required': True,
              'shape': [2, 12, 56, 56],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 24, 28, 28],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 48, 14, 14],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 1, 32, 32],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 2, 48, 48],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 4, 96, 96],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 8, 192, 192],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 16, 384, 384],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [8, 8, 64, 64],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'attn_weights',
              'required': True,
              'shape': [8, 12, 128, 128],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'attn_weights',
              'required': True,
              'shape': [8, 16, 256, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 32, 128, 128],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 64, 64, 64],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'attn_weights',
              'required': True,
              'shape': [2, 32, 256, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 1, 1, 1],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 1, 8, 8],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 1, 16, 16],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'attn_weights',
              'required': True,
              'shape': [1, 4, 128, 128],
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
