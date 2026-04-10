"""Block-level TileLang design for concat_dv2 limited to concat_dim == 0.

This design mirrors the effective semantics of the Ascend C implementation:
each input is treated as a 2D tensor of shape [dim0_i, sameDimSize], and the
output is formed by concatenating rows along dim0. Each block owns a contiguous
row tile of the output and resolves which input tensor supplies each row.
Fine-grained UB copy details are intentionally left to the tile-level file.
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

            with T.Scope("V"):
                for r in T.serial(sub_block_m):
                    row_idx = row_base + r
                    if row_idx < total_m:
                        # TODO(tile-level):
                        # - load X0[row_idx, :] into UB
                        # - store the row to Y[row_idx, :]
                        _ = X0
                        _ = Y
                        _ = row_idx

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

            with T.Scope("V"):
                for r in T.serial(sub_block_m):
                    row_idx = row_base + r
                    if row_idx < total_m:
                        # TODO(tile-level):
                        # - if row_idx < M0, read X0[row_idx, :]
                        # - else read X1[row_idx - M0, :]
                        # - store the selected row into Y[row_idx, :]
                        _ = X0
                        _ = X1
                        _ = Y
                        _ = row_idx

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def concat_dim0_3(M0, M1, M2, N, dtype="float32"):
    total_m = M0 + M1 + M2
    block_m = _select_block_m(total_m)
    vec_num = 2
    sub_block_m = block_m // vec_num
    block_num = (total_m + block_m - 1) // block_m

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

            with T.Scope("V"):
                for r in T.serial(sub_block_m):
                    row_idx = row_base + r
                    if row_idx < total_m:
                        # TODO(tile-level):
                        # - if row_idx < M0, read X0[row_idx, :]
                        # - elif row_idx < M0 + M1, read X1[row_idx - M0, :]
                        # - else read X2[row_idx - M0 - M1, :]
                        # - store the selected row into Y[row_idx, :]
                        _ = X0
                        _ = X1
                        _ = X2
                        _ = Y
                        _ = row_idx

    return main


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def concat_dim0_4(M0, M1, M2, M3, N, dtype="float32"):
    total_m = M0 + M1 + M2 + M3
    block_m = _select_block_m(total_m)
    vec_num = 2
    sub_block_m = block_m // vec_num
    block_num = (total_m + block_m - 1) // block_m

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

            with T.Scope("V"):
                for r in T.serial(sub_block_m):
                    row_idx = row_base + r
                    if row_idx < total_m:
                        # TODO(tile-level):
                        # - if row_idx < M0, read X0[row_idx, :]
                        # - elif row_idx < M0 + M1, read X1[row_idx - M0, :]
                        # - elif row_idx < M0 + M1 + M2, read X2[row_idx - M0 - M1, :]
                        # - else read X3[row_idx - M0 - M1 - M2, :]
                        # - store the selected row into Y[row_idx, :]
                        _ = X0
                        _ = X1
                        _ = X2
                        _ = X3
                        _ = Y
                        _ = row_idx

    return main
