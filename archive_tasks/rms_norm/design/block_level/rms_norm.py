"""Block-level TileLang design for RMSNorm.

This file mirrors the tile-level persistent-kernel scheduling, but keeps only
the coarse-grained block structure and vector-side pipeline skeleton.
Fine-grained compute details are intentionally left as TODO comments.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def rms_norm(M, N, eps=1e-5, dtype="float32"):
    block_M = 64
    num_physical_cores = 20
    assert M % block_M == 0
    m_num = M // block_M
    assert m_num > 0
    usedCoreNum = min(num_physical_cores, m_num)
    tasksPerCore = (m_num + usedCoreNum - 1) // usedCoreNum
    vec_num = 2
    sub_block_M = block_M // vec_num

    row_factor = 8
    row_loops = sub_block_M // row_factor

    @T.prim_func
    def merge_n(
        X: T.Tensor((M, N), dtype),
        Gamma: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            coreIdx = cid

            # Block-level persistent-kernel pipeline:
            #   1) Fixed physical cores iterate over a contiguous row-block range.
            #   2) Each AIV sub-block handles one half-block of rows.
            #   3) merge_n processes row_factor rows together inside each sub-block.
            for localIdx in T.serial(tasksPerCore):
                bx = coreIdx * tasksPerCore + localIdx

                with T.Scope("V"):
                    if bx < m_num:
                        for r in T.serial(row_loops):
                            row_base = bx * block_M + vid * sub_block_M + r * row_factor

                            # TODO(tile-level):
                            # - load row_factor rows from X[row_base:row_base + row_factor, :]
                            # - load/broadcast Gamma once per core or per task as needed
                            # - compute row-wise sum(x^2), divide by N, add eps, and rsqrt
                            # - broadcast rstd to width N
                            # - apply RMSNorm: out = x * rstd * gamma
                            # - store the normalized rows back to Y
                            _ = X
                            _ = Gamma
                            _ = Y
                            _ = row_base

    @T.prim_func
    def single_row(
        X: T.Tensor((M, N), dtype),
        Gamma: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(usedCoreNum, is_npu=True) as (cid, vid):
            coreIdx = cid

            # Block-level persistent-kernel pipeline:
            #   1) Fixed physical cores iterate over a contiguous row-block range.
            #   2) Each AIV sub-block handles one half-block of rows.
            #   3) single_row processes one row at a time inside each sub-block.
            for localIdx in T.serial(tasksPerCore):
                bx = coreIdx * tasksPerCore + localIdx

                with T.Scope("V"):
                    if bx < m_num:
                        for row in T.serial(sub_block_M):
                            row_idx = bx * block_M + vid * sub_block_M + row

                            # TODO(tile-level):
                            # - load one row from X[row_idx, :]
                            # - load Gamma
                            # - compute row-wise sum(x^2), divide by N, add eps, and rsqrt
                            # - broadcast inv_rms to width N
                            # - apply RMSNorm: out = x * inv_rms * gamma
                            # - store the normalized row back to Y
                            _ = X
                            _ = Gamma
                            _ = Y
                            _ = row_idx

    _ = eps

    if N <= 1024:
        return merge_n
    return single_row
