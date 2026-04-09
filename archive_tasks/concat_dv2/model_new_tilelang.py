import torch
import torch.nn as nn

from concat_dv2.design.tile_level.concat_dim0 import (
    concat_dim0_1 as tl_concat_dim0_1,
    concat_dim0_2 as tl_concat_dim0_2,
    concat_dim0_3 as tl_concat_dim0_3,
    concat_dim0_4 as tl_concat_dim0_4,
)


SUPPORTED_DTYPES = (
    torch.float16,
    torch.float32,
    torch.bfloat16,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._kernel_cache = {}

    def _normalize_inputs(self, inputs, concat_dim: int):
        if not isinstance(inputs, (list, tuple)) or not inputs:
            raise ValueError("inputs must be a non-empty list or tuple of tensors")
        if concat_dim != 0:
            raise ValueError(f"concat_dv2 TileLang implementation only supports concat_dim=0, got {concat_dim}")
        if len(inputs) > 4:
            raise ValueError(f"up to 4 inputs are supported, got {len(inputs)}")

        reference = inputs[0]
        if not isinstance(reference, torch.Tensor):
            raise TypeError("all inputs must be torch.Tensor")
        if reference.ndim < 1:
            raise ValueError("each input tensor must have rank >= 1")
        if reference.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype: {reference.dtype}")

        tail_shape = tuple(int(dim) for dim in reference.shape[1:])
        rank = reference.ndim
        dtype = reference.dtype

        normalized = []
        dim0_sizes = []
        for tensor in inputs:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("all inputs must be torch.Tensor")
            if tensor.ndim != rank:
                raise ValueError("all inputs must have the same rank")
            if tensor.dtype != dtype:
                raise ValueError("all inputs must have the same dtype")
            if tuple(int(dim) for dim in tensor.shape[1:]) != tail_shape:
                raise ValueError("all non-concat dimensions must match")

            dim0 = int(tensor.shape[0])
            normalized.append(tensor.contiguous().reshape(dim0, -1))
            dim0_sizes.append(dim0)

        return normalized, tail_shape, dim0_sizes

    def _build_kernel(self, inputs_2d):
        dim0_sizes = [int(tensor.shape[0]) for tensor in inputs_2d]
        same_dim_size = int(inputs_2d[0].shape[1])
        dtype = str(inputs_2d[0].dtype).split(".")[-1]

        cache_key = (tuple(dim0_sizes), same_dim_size, dtype)
        cached = self._kernel_cache.get(cache_key)
        if cached is not None:
            return cached

        input_count = len(inputs_2d)
        if input_count == 1:
            kernel = tl_concat_dim0_1(dim0_sizes[0], same_dim_size, dtype=dtype)
        elif input_count == 2:
            kernel = tl_concat_dim0_2(dim0_sizes[0], dim0_sizes[1], same_dim_size, dtype=dtype)
        elif input_count == 3:
            kernel = tl_concat_dim0_3(dim0_sizes[0], dim0_sizes[1], dim0_sizes[2], same_dim_size, dtype=dtype)
        elif input_count == 4:
            kernel = tl_concat_dim0_4(
                dim0_sizes[0],
                dim0_sizes[1],
                dim0_sizes[2],
                dim0_sizes[3],
                same_dim_size,
                dtype=dtype,
            )
        else:
            raise ValueError(f"unsupported input count: {input_count}")

        self._kernel_cache[cache_key] = kernel
        return kernel

    def _invoke_kernel(self, kernel, inputs_2d):
        input_count = len(inputs_2d)
        if input_count == 1:
            return kernel(inputs_2d[0])
        if input_count == 2:
            return kernel(inputs_2d[0], inputs_2d[1])
        if input_count == 3:
            return kernel(inputs_2d[0], inputs_2d[1], inputs_2d[2])
        return kernel(inputs_2d[0], inputs_2d[1], inputs_2d[2], inputs_2d[3])

    def _warmup_if_needed(self, cache_key, kernel, inputs_2d):
        warmed_key = ("warmed", cache_key)
        if self._kernel_cache.get(warmed_key):
            return
        warm_inputs = [tensor.clone() for tensor in inputs_2d]
        _ = self._invoke_kernel(kernel, warm_inputs)
        self._kernel_cache[warmed_key] = True

    def forward(self, inputs, concat_dim: int = 0):
        inputs_2d, tail_shape, dim0_sizes = self._normalize_inputs(inputs, concat_dim)
        cache_key = (
            tuple(int(tensor.shape[0]) for tensor in inputs_2d),
            int(inputs_2d[0].shape[1]),
            str(inputs_2d[0].dtype).split(".")[-1],
        )
        kernel = self._build_kernel(inputs_2d)
        self._warmup_if_needed(cache_key, kernel, inputs_2d)
        y_2d = self._invoke_kernel(kernel, inputs_2d)

        output_shape = (sum(dim0_sizes), *tail_shape)
        return y_2d.reshape(output_shape)
