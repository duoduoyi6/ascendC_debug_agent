"""Unified TileLang design for RMSNorm.

This file keeps three specialized prim_funcs:
- merge_n: preferred when N <= 1024
- single_row: preferred when 1024 < N <= 8192
- splitd: preferred when N > 8192

The kernel performs both the input cast to float32 and the final cast back to
the requested output dtype, so the Python wrapper only handles reshaping.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def rms_norm(M, N, eps=1e-5, in_dtype="float32", out_dtype="float32"):
    block_M = 64
    block_N = 1024
    num_physical_cores = 20
    m_num = T.ceildiv(M, block_M)
    used_core_num = min(num_physical_cores, m_num)
    tasks_per_core = T.ceildiv(m_num, used_core_num)
    vec_num = 2
    sub_block_M = block_M // vec_num

    row_factor = 8
    row_loops = T.ceildiv(sub_block_M, row_factor)
    n_num = T.ceildiv(N, block_N)
    need_cast_in = in_dtype != "float32"
    need_cast_out = out_dtype != "float32"
    out_cast_mode = "CAST_ROUND" if out_dtype == "bfloat16" else "CAST_NONE"

    eps_const = T.float32(eps)
    inv_n_const = T.float32(1.0 / N)

    @T.prim_func
    def merge_n(
        X: T.Tensor((M, N), in_dtype),
        Gamma: T.Tensor((N,), in_dtype),
        Y: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            x_in_ub = T.alloc_ub((row_factor, N), in_dtype)
            gamma_in_ub = T.alloc_ub((1, N), in_dtype)
            out_cast_ub = T.alloc_ub((row_factor, N), out_dtype)
            single_x_in_ub = T.alloc_ub((1, N), in_dtype)
            single_out_cast_ub = T.alloc_ub((1, N), out_dtype)
            x_ub = T.alloc_ub((row_factor, N), "float32")
            x_sq_ub = T.alloc_ub((row_factor, N), "float32")
            gamma_ub = T.alloc_ub((1, N), "float32")
            gamma_broad_ub = T.alloc_ub((row_factor, N), "float32")
            sum_sq_ub = T.alloc_ub((row_factor, 1), "float32")
            rstd_ub = T.alloc_ub((row_factor, 1), "float32")
            rstd_broad_ub = T.alloc_ub((row_factor, N), "float32")
            out_ub = T.alloc_ub((row_factor, N), "float32")

            single_x_ub = T.alloc_ub((1, N), "float32")
            single_x_sq_ub = T.alloc_ub((1, N), "float32")
            single_sum_sq_ub = T.alloc_ub((1, 1), "float32")
            single_rstd_ub = T.alloc_ub((1, 1), "float32")
            single_rstd_broad_ub = T.alloc_ub((1, N), "float32")
            single_out_ub = T.alloc_ub((1, N), "float32")

            inv_n_ub = T.alloc_ub((row_factor, 1), "float32")
            eps_ub = T.alloc_ub((row_factor, 1), "float32")
            single_inv_n_ub = T.alloc_ub((1, 1), "float32")
            single_eps_ub = T.alloc_ub((1, 1), "float32")

            reduce_tmp = T.alloc_ub((2 * row_factor * N,), "uint8")
            gamma_bcast_tmp = T.alloc_ub((2 * row_factor, N), "uint8")
            rstd_bcast_tmp = T.alloc_ub((2 * row_factor, N), "uint8")
            single_reduce_tmp = T.alloc_ub((2 * N,), "uint8")
            single_bcast_tmp = T.alloc_ub((2, N), "uint8")

            with T.Scope("V"):
                if need_cast_in:
                    T.copy(Gamma[0], gamma_in_ub)
                    T.tile.cast(gamma_ub, gamma_in_ub, mode="CAST_NONE", count=N)
                else:
                    T.copy(Gamma[0], gamma_ub)
                T.tile.broadcast(gamma_broad_ub, gamma_ub, gamma_bcast_tmp)
                T.tile.fill(inv_n_ub, inv_n_const)
                T.tile.fill(eps_ub, eps_const)
                T.tile.fill(single_inv_n_ub, inv_n_const)
                T.tile.fill(single_eps_ub, eps_const)

                for local_idx in T.serial(tasks_per_core):
                    bx = cid * tasks_per_core + local_idx
                    if bx < m_num:
                        for r in T.serial(row_loops):
                            row_base = bx * block_M + vid * sub_block_M + r * row_factor
                            if row_base + row_factor <= M:
                                if need_cast_in:
                                    T.copy(X[row_base:row_base + row_factor, :], x_in_ub)
                                    T.tile.cast(x_ub, x_in_ub, mode="CAST_NONE", count=row_factor * N)
                                else:
                                    T.copy(X[row_base:row_base + row_factor, :], x_ub)
                                T.tile.mul(x_sq_ub, x_ub, x_ub)
                                T.reduce_sum(x_sq_ub, sum_sq_ub, reduce_tmp, dim=-1)
                                T.tile.mul(sum_sq_ub, sum_sq_ub, inv_n_ub)
                                T.tile.add(sum_sq_ub, sum_sq_ub, eps_ub)
                                T.tile.rsqrt(rstd_ub, sum_sq_ub)
                                T.tile.broadcast(rstd_broad_ub, rstd_ub, rstd_bcast_tmp)
                                T.tile.mul(out_ub, x_ub, rstd_broad_ub)
                                T.tile.mul(out_ub, out_ub, gamma_broad_ub)
                                if need_cast_out:
                                    T.tile.cast(out_cast_ub, out_ub, mode=out_cast_mode, count=row_factor * N)
                                    T.copy(out_cast_ub, Y[row_base:row_base + row_factor, :])
                                else:
                                    T.copy(out_ub, Y[row_base:row_base + row_factor, :])
                            else:
                                for rr in T.serial(row_factor):
                                    row_idx = row_base + rr
                                    if row_idx < M:
                                        if need_cast_in:
                                            T.copy(X[row_idx, :], single_x_in_ub)
                                            T.tile.cast(single_x_ub, single_x_in_ub, mode="CAST_NONE", count=N)
                                        else:
                                            T.copy(X[row_idx, :], single_x_ub)
                                        T.tile.mul(single_x_sq_ub, single_x_ub, single_x_ub)
                                        T.reduce_sum(single_x_sq_ub, single_sum_sq_ub, single_reduce_tmp, dim=-1)
                                        T.tile.mul(single_sum_sq_ub, single_sum_sq_ub, single_inv_n_ub)
                                        T.tile.add(single_sum_sq_ub, single_sum_sq_ub, single_eps_ub)
                                        T.tile.rsqrt(single_rstd_ub, single_sum_sq_ub)
                                        T.tile.broadcast(single_rstd_broad_ub, single_rstd_ub, single_bcast_tmp)
                                        T.tile.mul(single_out_ub, single_x_ub, single_rstd_broad_ub)
                                        T.tile.mul(single_out_ub, single_out_ub, gamma_ub)
                                        if need_cast_out:
                                            T.tile.cast(single_out_cast_ub, single_out_ub, mode=out_cast_mode, count=N)
                                            T.copy(single_out_cast_ub, Y[row_idx, :])
                                        else:
                                            T.copy(single_out_ub, Y[row_idx, :])

    @T.prim_func
    def single_row(
        X: T.Tensor((M, N), in_dtype),
        Gamma: T.Tensor((N,), in_dtype),
        Y: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            x_in_ub = T.alloc_ub((1, N), in_dtype)
            gamma_in_ub = T.alloc_ub((1, N), in_dtype)
            out_cast_ub = T.alloc_ub((1, N), out_dtype)
            x_ub = T.alloc_ub((1, N), "float32")
            x_sq_ub = T.alloc_ub((1, N), "float32")
            gamma_ub = T.alloc_ub((1, N), "float32")
            sum_sq_ub = T.alloc_ub((1, 1), "float32")
            inv_rms_ub = T.alloc_ub((1, 1), "float32")
            inv_rms_broad_ub = T.alloc_ub((1, N), "float32")
            out_ub = T.alloc_ub((1, N), "float32")
            inv_n_ub = T.alloc_ub((1, 1), "float32")
            eps_ub = T.alloc_ub((1, 1), "float32")

            reduce_tmp = T.alloc_ub((2 * N,), "uint8")
            broadcast_tmp = T.alloc_ub((2, N), "uint8")

            with T.Scope("V"):
                if need_cast_in:
                    T.copy(Gamma[0], gamma_in_ub)
                    T.tile.cast(gamma_ub, gamma_in_ub, mode="CAST_NONE", count=N)
                else:
                    T.copy(Gamma[0], gamma_ub)
                T.tile.fill(inv_n_ub, inv_n_const)
                T.tile.fill(eps_ub, eps_const)

                for local_idx in T.serial(tasks_per_core):
                    bx = cid * tasks_per_core + local_idx
                    if bx < m_num:
                        for row in T.serial(sub_block_M):
                            row_idx = bx * block_M + vid * sub_block_M + row
                            if row_idx < M:
                                if need_cast_in:
                                    T.copy(X[row_idx, :], x_in_ub)
                                    T.tile.cast(x_ub, x_in_ub, mode="CAST_NONE", count=N)
                                else:
                                    T.copy(X[row_idx, :], x_ub)
                                T.tile.mul(x_sq_ub, x_ub, x_ub)
                                T.reduce_sum(x_sq_ub, sum_sq_ub, reduce_tmp, dim=-1)
                                T.tile.mul(sum_sq_ub, sum_sq_ub, inv_n_ub)
                                T.tile.add(sum_sq_ub, sum_sq_ub, eps_ub)
                                T.tile.rsqrt(inv_rms_ub, sum_sq_ub)
                                T.tile.broadcast(inv_rms_broad_ub, inv_rms_ub, broadcast_tmp)
                                T.tile.mul(out_ub, x_ub, inv_rms_broad_ub)
                                T.tile.mul(out_ub, out_ub, gamma_ub)
                                if need_cast_out:
                                    T.tile.cast(out_cast_ub, out_ub, mode=out_cast_mode, count=N)
                                    T.copy(out_cast_ub, Y[row_idx, :])
                                else:
                                    T.copy(out_ub, Y[row_idx, :])

    @T.prim_func
    def splitd(
        X: T.Tensor((M, N), in_dtype),
        Gamma: T.Tensor((N,), in_dtype),
        Y: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            x_in_ub = T.alloc_ub((1, block_N), in_dtype)
            gamma_in_ub = T.alloc_ub((1, block_N), in_dtype)
            x_ub = T.alloc_ub((1, block_N), "float32")
            x_sq_ub = T.alloc_ub((1, block_N), "float32")
            gamma_ub = T.alloc_ub((1, block_N), "float32")
            gamma_broad_ub = T.alloc_ub((1, block_N), "float32")
            out_ub = T.alloc_ub((1, block_N), "float32")
            out_cast_ub = T.alloc_ub((1, block_N), out_dtype)
            sum_sq_ub = T.alloc_ub((1, 1), "float32")
            partial_sum_ub = T.alloc_ub((1, 1), "float32")
            inv_rms_ub = T.alloc_ub((1, 1), "float32")
            inv_rms_broad_ub = T.alloc_ub((1, block_N), "float32")
            inv_n_ub = T.alloc_ub((1, 1), "float32")
            eps_ub = T.alloc_ub((1, 1), "float32")

            reduce_tmp = T.alloc_ub((2 * block_N,), "uint8")
            inv_rms_bcast_tmp = T.alloc_ub((2, block_N), "uint8")
            gamma_bcast_tmp = T.alloc_ub((2, block_N), "uint8")

            with T.Scope("V"):
                T.tile.fill(inv_n_ub, inv_n_const)
                T.tile.fill(eps_ub, eps_const)

                for local_idx in T.serial(tasks_per_core):
                    bx = cid * tasks_per_core + local_idx
                    if bx < m_num:
                        for row in T.serial(sub_block_M):
                            row_idx = bx * block_M + vid * sub_block_M + row
                            if row_idx < M:
                                T.tile.fill(sum_sq_ub, 0.0)

                                for by in T.serial(n_num):
                                    col_base = by * block_N
                                    T.tile.fill(x_ub, 0.0)
                                    if need_cast_in:
                                        T.tile.fill(x_in_ub, 0.0)
                                        if col_base + block_N <= N:
                                            T.copy(X[row_idx:row_idx + 1, col_base:col_base + block_N], x_in_ub)
                                        else:
                                            T.copy(X[row_idx:row_idx + 1, col_base:N], x_in_ub[:, 0:N - col_base])
                                        T.tile.cast(x_ub, x_in_ub, mode="CAST_NONE", count=block_N)
                                    else:
                                        if col_base + block_N <= N:
                                            T.copy(X[row_idx:row_idx + 1, col_base:col_base + block_N], x_ub)
                                        else:
                                            T.copy(X[row_idx:row_idx + 1, col_base:N], x_ub[:, 0:N - col_base])

                                    T.tile.mul(x_sq_ub, x_ub, x_ub)
                                    T.reduce_sum(x_sq_ub, partial_sum_ub, reduce_tmp, dim=-1)
                                    T.tile.add(sum_sq_ub, sum_sq_ub, partial_sum_ub)

                                T.tile.mul(sum_sq_ub, sum_sq_ub, inv_n_ub)
                                T.tile.add(sum_sq_ub, sum_sq_ub, eps_ub)
                                T.tile.rsqrt(inv_rms_ub, sum_sq_ub)

                                for by in T.serial(n_num):
                                    col_base = by * block_N
                                    T.tile.fill(x_ub, 0.0)
                                    if need_cast_in:
                                        T.tile.fill(x_in_ub, 0.0)
                                        if col_base + block_N <= N:
                                            T.copy(X[row_idx:row_idx + 1, col_base:col_base + block_N], x_in_ub)
                                        else:
                                            T.copy(X[row_idx:row_idx + 1, col_base:N], x_in_ub[:, 0:N - col_base])
                                        T.tile.cast(x_ub, x_in_ub, mode="CAST_NONE", count=block_N)
                                    else:
                                        if col_base + block_N <= N:
                                            T.copy(X[row_idx:row_idx + 1, col_base:col_base + block_N], x_ub)
                                        else:
                                            T.copy(X[row_idx:row_idx + 1, col_base:N], x_ub[:, 0:N - col_base])

                                    T.tile.fill(gamma_ub, 0.0)
                                    if need_cast_in:
                                        T.tile.fill(gamma_in_ub, 0.0)
                                        if col_base + block_N <= N:
                                            T.copy(Gamma[col_base:col_base + block_N], gamma_in_ub[0, :])
                                        else:
                                            T.copy(Gamma[col_base:N], gamma_in_ub[0, 0:N - col_base])
                                        T.tile.cast(gamma_ub, gamma_in_ub, mode="CAST_NONE", count=block_N)
                                    else:
                                        if col_base + block_N <= N:
                                            T.copy(Gamma[col_base:col_base + block_N], gamma_ub[0, :])
                                        else:
                                            T.copy(Gamma[col_base:N], gamma_ub[0, 0:N - col_base])

                                    T.tile.broadcast(inv_rms_broad_ub, inv_rms_ub, inv_rms_bcast_tmp)
                                    T.tile.broadcast(gamma_broad_ub, gamma_ub, gamma_bcast_tmp)
                                    T.tile.mul(out_ub, x_ub, inv_rms_broad_ub)
                                    T.tile.mul(out_ub, out_ub, gamma_broad_ub)

                                    if col_base + block_N <= N:
                                        if need_cast_out:
                                            T.tile.cast(out_cast_ub, out_ub, mode=out_cast_mode, count=block_N)
                                            T.copy(out_cast_ub, Y[row_idx:row_idx + 1, col_base:col_base + block_N])
                                        else:
                                            T.copy(out_ub, Y[row_idx:row_idx + 1, col_base:col_base + block_N])
                                    else:
                                        if need_cast_out:
                                            T.tile.cast(out_cast_ub, out_ub, mode=out_cast_mode, count=block_N)
                                            T.copy(out_cast_ub[:, 0:N - col_base], Y[row_idx:row_idx + 1, col_base:N])
                                        else:
                                            T.copy(out_ub[:, 0:N - col_base], Y[row_idx:row_idx + 1, col_base:N])

    if N <= 1024:
        return merge_n
    if N > 8192:
        return splitd
    return single_row
