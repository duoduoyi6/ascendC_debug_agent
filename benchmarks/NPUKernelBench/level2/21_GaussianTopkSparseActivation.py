import torch
import torch.nn as nn
import math

class Model(nn.Module):
    """
    Gaussian-based Top-k Sparse Activation Module.

    Computes adaptive sparsity threshold based on input statistics:
    1. Compute mean and std of input across feature dimension
    2. Calculate threshold = mean + std * norm.icdf(target_sparsity)
    3. Apply ReLU(input - threshold) to create sparse activations
    """
    def __init__(self):
        super(Model, self).__init__()
        self.a1 = -3.969683028665376e+01
        self.a2 = 2.209460984245205e+02
        self.a3 = -2.759285104469687e+02
        self.a4 = 1.383577518672690e+02
        self.a5 = -3.066479806614716e+01
        self.a6 = 2.506628277459239e+00
        self.b1 = -5.447609879822406e+01
        self.b2 = 1.615858368580409e+02
        self.b3 = -1.556989798598866e+02
        self.b4 = 6.680131188771972e+01
        self.b5 = -1.328068155288572e+01
        self.c1 = -7.784894002430293e-03
        self.c2 = -3.223964580411365e-01
        self.c3 = -2.400758277161838e+00
        self.c4 = -2.549732539343734e+00
        self.c5 = 4.374664141464968e+00
        self.c6 = 2.938163982698783e+00
        self.d1 = 7.784695709041462e-03
        self.d2 = 3.224671290700398e-01
        self.d3 = 2.445134137142996e+00
        self.d4 = 3.754408661907416e+00

    def _ndtri(self, p: torch.Tensor) -> torch.Tensor:
        """
        Inverse of the standard normal CDF (quantile function).

        Uses Abramowitz and Stegun approximation (formula 26.2.23).
        This is a rational approximation that works well for p in (0, 1).

        Args:
            p: Probability values in range (0, 1)

        Returns:
            Quantile values (z-scores) from standard normal distribution
        """
        p_low = 0.02425
        p_high = 1.0 - p_low
        result = torch.zeros_like(p)
        mask_low = p < p_low
        if mask_low.any():
            q = torch.sqrt(-2.0 * torch.log(p[mask_low]))
            result[mask_low] = (((((self.c1*q + self.c2)*q + self.c3)*q + self.c4)*q + self.c5)*q + self.c6) / ((((self.d1*q + self.d2)*q + self.d3)*q + self.d4)*q + 1.0)
        mask_mid = (p >= p_low) & (p <= p_high)
        if mask_mid.any():
            q = p[mask_mid] - 0.5
            r = q * q
            result[mask_mid] = (((((self.a1*r + self.a2)*r + self.a3)*r + self.a4)*r + self.a5)*r + self.a6)*q / (((((self.b1*r + self.b2)*r + self.b3)*r + self.b4)*r + self.b5)*r + 1.0)
        mask_high = p > p_high
        if mask_high.any():
            q = torch.sqrt(-2.0 * torch.log(1.0 - p[mask_high]))
            result[mask_high] = -(((((self.c1*q + self.c2)*q + self.c3)*q + self.c4)*q + self.c5)*q + self.c6) / ((((self.d1*q + self.d2)*q + self.d3)*q + self.d4)*q + 1.0)
        return result

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Gaussian-based top-k sparse activation.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, intermediate_size]
            target_sparsity: Float in [0, 1] indicating target sparsity level.
                            0.0 means no sparsity (all activations pass through).

        Returns:
            Sparsified tensor of same shape as input.
        """
        if target_sparsity == 0.0:
            return inputs
        input_f32 = inputs.to(torch.float32)
        inputs_mean = torch.mean(input_f32, dim=-1, keepdim=True)
        inputs_std = torch.std(input_f32, dim=-1, keepdim=True, unbiased=False)
        threshold = inputs_mean + inputs_std * self._ndtri(torch.tensor(target_sparsity, dtype=torch.float32, device=inputs.device))
        output = torch.nn.functional.relu(input_f32 - threshold)
        return output.to(torch.bfloat16)

INPUT_CASES = [{'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [1, 128, 512],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.5}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [2, 256, 1024],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.3}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [4, 512, 2048],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.7}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'inputs',
              'required': True,
              'shape': [1, 64, 256],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.25}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'inputs',
              'required': True,
              'shape': [2, 128, 512],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.75}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'inputs',
              'required': True,
              'shape': [1, 32, 128],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.1}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'inputs',
              'required': True,
              'shape': [2, 64, 256],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.9}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [1, 16, 64],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.0}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [1, 32, 128],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.01}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [1, 64, 256],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.99}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'inputs',
              'required': True,
              'shape': [8, 128, 512],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.5}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'inputs',
              'required': True,
              'shape': [16, 256, 1024],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.4}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'inputs',
              'required': True,
              'shape': [4, 64, 256],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.6}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'inputs',
              'required': True,
              'shape': [8, 128, 512],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.2}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [1, 512, 4096],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.8}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [2, 1024, 8192],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.35}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'inputs',
              'required': True,
              'shape': [1, 256, 2048],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.65}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'inputs',
              'required': True,
              'shape': [3, 128, 512],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.45}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'inputs',
              'required': True,
              'shape': [5, 64, 256],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.55}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'inputs',
              'required': True,
              'shape': [7, 32, 128],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.15}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [1, 7, 64],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.85}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [2, 13, 128],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.05}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [3, 17, 256],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.95}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'inputs',
              'required': True,
              'shape': [1, 11, 512],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.33}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'inputs',
              'required': True,
              'shape': [2, 19, 1024],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.67}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'inputs',
              'required': True,
              'shape': [1, 23, 512],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.22}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'inputs',
              'required': True,
              'shape': [2, 29, 256],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.78}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [1, 100, 512],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.5}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [2, 200, 1024],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.3}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'inputs',
              'required': True,
              'shape': [1, 150, 768],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.7}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'inputs',
              'required': True,
              'shape': [4, 300, 1536],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.4}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'inputs',
              'required': True,
              'shape': [2, 250, 1280],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.6}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'inputs',
              'required': True,
              'shape': [1, 175, 896],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.25}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [1, 1, 64],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.5}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [1, 1, 128],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.0}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'inputs',
              'required': True,
              'shape': [1, 2, 256],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.75}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'inputs',
              'required': True,
              'shape': [1, 3, 512],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.1}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'inputs',
              'required': True,
              'shape': [1, 4, 1024],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.9}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'inputs',
              'required': True,
              'shape': [1, 5, 2048],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.15}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [32, 128, 512],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.5}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [64, 64, 256],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.3}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'inputs',
              'required': True,
              'shape': [16, 256, 1024],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.7}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'inputs',
              'required': True,
              'shape': [24, 128, 512],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.4}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'inputs',
              'required': True,
              'shape': [12, 64, 256],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.6}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'inputs',
              'required': True,
              'shape': [20, 32, 128],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.2}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [1, 512, 14336],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.5}]},
 {'inputs': [{'dtype': 'float32',
              'name': 'inputs',
              'required': True,
              'shape': [2, 256, 11008],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.35}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'inputs',
              'required': True,
              'shape': [1, 128, 6144],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.65}]},
 {'inputs': [{'dtype': 'float16',
              'name': 'inputs',
              'required': True,
              'shape': [4, 64, 3072],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.45}]},
 {'inputs': [{'dtype': 'bfloat16',
              'name': 'inputs',
              'required': True,
              'shape': [2, 32, 1536],
              'type': 'tensor'},
             {'dtype': 'float',
              'name': 'target_sparsity',
              'required': True,
              'type': 'attr',
              'value': 0.55}]}]

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
