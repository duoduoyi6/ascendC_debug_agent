import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.to(torch.float32)
        mean_sq = torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_sq + self.eps)
        out = x_fp32 * inv_rms * gamma.to(torch.float32)
        return out.to(x.dtype)


RMS_NORM_CASES = [
    {"shape": [1024, 1024], "dtype": torch.float32, "x_mode": "randn", "gamma_mode": "randn"},
    {"shape": [128, 4096], "dtype": torch.float16, "x_mode": "randn", "gamma_mode": "randn"},
    {"shape": [64, 3584], "dtype": torch.bfloat16, "x_mode": "randn", "gamma_mode": "randn"},
    {"shape": [1, 32, 1024], "dtype": torch.float32, "x_mode": "randn", "gamma_mode": "ones"},
    {"shape": [2, 16, 2048], "dtype": torch.float16, "x_mode": "randn", "gamma_mode": "randn"},
    {"shape": [2, 8, 16, 256], "dtype": torch.float32, "x_mode": "randn", "gamma_mode": "randn"},
    {"shape": [17, 1536], "dtype": torch.float32, "x_mode": "randn", "gamma_mode": "randn"},
    {"shape": [4, 513], "dtype": torch.float32, "x_mode": "small", "gamma_mode": "randn"},
    {"shape": [8, 1024], "dtype": torch.float32, "x_mode": "zeros", "gamma_mode": "randn"},
]


def _make_tensor(shape, dtype, mode, seed):
    generator = torch.Generator().manual_seed(seed)
    if mode == "randn":
        return torch.randn(*shape, dtype=dtype, generator=generator)
    if mode == "small":
        return torch.randn(*shape, dtype=dtype, generator=generator) * 1e-4
    if mode == "zeros":
        return torch.zeros(*shape, dtype=dtype)
    if mode == "ones":
        return torch.ones(*shape, dtype=dtype)
    raise ValueError(f"Unsupported tensor mode: {mode}")


def get_input_groups():
    input_groups = []
    for idx, case in enumerate(RMS_NORM_CASES):
        shape = case["shape"]
        dtype = case["dtype"]
        hidden_size = shape[-1]
        x = _make_tensor(shape, dtype, case["x_mode"], seed=2026 + idx)
        gamma = _make_tensor([hidden_size], dtype, case["gamma_mode"], seed=3026 + idx)
        input_groups.append([x, gamma])
    return input_groups


def get_init_inputs():
    return []
