import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused FFT size padding and real FFT computation for Hyena convolution.
    Pads input to 2*seqlen for circular convolution, computes real FFT (rfft),
    and normalizes by fft_size.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Applies FFT size padding and real FFT computation for Hyena convolution.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, channels, seqlen].
                              Supports float32.

        Returns:
            tuple: (x_freq_real, x_freq_imag)
                - x_freq_real (torch.Tensor): Real part of normalized frequency domain output
                                              with shape [batch_size, channels, seqlen+1].
                - x_freq_imag (torch.Tensor): Imaginary part of normalized frequency domain output
                                              with shape [batch_size, channels, seqlen+1].
        """
        batch, channels, seqlen = x.shape
        fft_size = 2 * seqlen

        x_f32 = x.to(torch.float32)

        x_freq = torch.fft.rfft(x_f32, n=fft_size)

        x_freq = x_freq / fft_size

        x_freq_real = x_freq.real.contiguous()
        x_freq_imag = x_freq.imag.contiguous()

        return x_freq_real, x_freq_imag

INPUT_CASES = [{'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 1, 16],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 1, 32],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 1, 64],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 1, 128],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 1, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 1, 512],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 1, 1024],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 1, 2048],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [2, 4, 128],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [4, 8, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [8, 16, 512],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [16, 32, 1024],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 64, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [2, 128, 512],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [4, 256, 1024],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 1, 8],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 1, 4],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 2, 16],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [2, 2, 32],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [4, 4, 64],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [1, 1, 128],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [2, 4, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [4, 8, 512],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'x',
              'required': True,
              'shape': [8, 16, 1024],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [1, 1, 64],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [2, 2, 128],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [4, 4, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'x',
              'required': True,
              'shape': [8, 8, 512],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 1, 100],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 1, 200],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [2, 3, 150],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [3, 5, 250],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [5, 7, 350],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [7, 11, 450],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 32, 128],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 64, 256],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 128, 512],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 256, 1024],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [2, 1, 512],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [4, 1, 1024],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [8, 1, 2048],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [16, 1, 4096],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 3, 224],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [2, 3, 224],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 64, 56],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [2, 64, 56],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 128, 28],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 256, 14],
              'type': 'tensor'}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'x',
              'required': True,
              'shape': [1, 512, 7],
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
