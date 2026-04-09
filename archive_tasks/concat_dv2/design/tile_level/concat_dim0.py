"""Tile-level TileLang implementation for concat_dv2 limited to concat_dim == 0.

The original Ascend C kernel partitions the flattened output into contiguous
segments and gathers the matching fragments from multiple inputs. In TileLang,
this implementation keeps the same effective dim0 semantics but uses a simpler
row-tiled vector schedule: each input is reshaped to [dim0_i, sameDimSize], and
blocks copy output rows from the matching source tensor with UB staging.
"""

import tilelang
import tilelang.language as T


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _select_block_m(total_m: int) -> int:
    for candidate in (64, 32, 16, 8, 4, 2):
        if total_m >= candidate:
            return candidate
    return 2


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def concat_dim0_1(M0, N, dtype="float32"):
    total_m = M0
    block_m = _select_block_m(total_m)
    vec_num = 2
    sub_block_m = block_m // vec_num
    block_num = (total_m + block_m - 1) // block_m

    @T.prim_func
    def main(
        X0: T.Tensor((M0, N), dtype),
        Y: T.Tensor((total_m, N), dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_m + vid * sub_block_m
            row_ub = T.alloc_ub((1, N), dtype)

            with T.Scope("V"):
                for r in T.serial(sub_block_m):
                    row_idx = row_base + r
                    if row_idx < total_m:
                        T.copy(X0[row_idx, :], row_ub)
                        T.copy(row_ub, Y[row_idx, :])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def concat_dim0_2(M0, M1, N, dtype="float32"):
    total_m = M0 + M1
    block_m = _select_block_m(total_m)
    vec_num = 2
    sub_block_m = block_m // vec_num
    block_num = (total_m + block_m - 1) // block_m

    @T.prim_func
    def main(
        X0: T.Tensor((M0, N), dtype),
        X1: T.Tensor((M1, N), dtype),
        Y: T.Tensor((total_m, N), dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_m + vid * sub_block_m
            row_ub = T.alloc_ub((1, N), dtype)

            with T.Scope("V"):
                for r in T.serial(sub_block_m):
                    row_idx = row_base + r
                    if row_idx < total_m:
                        if row_idx < M0:
                            T.copy(X0[row_idx, :], row_ub)
                        else:
                            T.copy(X1[row_idx - M0, :], row_ub)
                        T.copy(row_ub, Y[row_idx, :])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def concat_dim0_3(M0, M1, M2, N, dtype="float32"):
    total_m = M0 + M1 + M2
    block_m = _select_block_m(total_m)
    vec_num = 2
    sub_block_m = block_m // vec_num
    block_num = (total_m + block_m - 1) // block_m
    prefix01 = M0 + M1

    @T.prim_func
    def main(
        X0: T.Tensor((M0, N), dtype),
        X1: T.Tensor((M1, N), dtype),
        X2: T.Tensor((M2, N), dtype),
        Y: T.Tensor((total_m, N), dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_m + vid * sub_block_m
            row_ub = T.alloc_ub((1, N), dtype)

            with T.Scope("V"):
                for r in T.serial(sub_block_m):
                    row_idx = row_base + r
                    if row_idx < total_m:
                        if row_idx < M0:
                            T.copy(X0[row_idx, :], row_ub)
                        elif row_idx < prefix01:
                            T.copy(X1[row_idx - M0, :], row_ub)
                        else:
                            T.copy(X2[row_idx - prefix01, :], row_ub)
                        T.copy(row_ub, Y[row_idx, :])

    return main


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def concat_dim0_4(M0, M1, M2, M3, N, dtype="float32"):
    total_m = M0 + M1 + M2 + M3
    block_m = _select_block_m(total_m)
    vec_num = 2
    sub_block_m = block_m // vec_num
    block_num = (total_m + block_m - 1) // block_m
    prefix01 = M0 + M1
    prefix012 = M0 + M1 + M2

    @T.prim_func
    def main(
        X0: T.Tensor((M0, N), dtype),
        X1: T.Tensor((M1, N), dtype),
        X2: T.Tensor((M2, N), dtype),
        X3: T.Tensor((M3, N), dtype),
        Y: T.Tensor((total_m, N), dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_m + vid * sub_block_m
            row_ub = T.alloc_ub((1, N), dtype)

            with T.Scope("V"):
                for r in T.serial(sub_block_m):
                    row_idx = row_base + r
                    if row_idx < total_m:
                        if row_idx < M0:
                            T.copy(X0[row_idx, :], row_ub)
                        elif row_idx < prefix01:
                            T.copy(X1[row_idx - M0, :], row_ub)
                        elif row_idx < prefix012:
                            T.copy(X2[row_idx - prefix01, :], row_ub)
                        else:
                            T.copy(X3[row_idx - prefix012, :], row_ub)
                        T.copy(row_ub, Y[row_idx, :])

    return main
