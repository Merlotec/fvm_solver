"""Triton ELL-packed SpMM kernel and Python wrapper."""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_B': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_B': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_B': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_B': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_B': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_B': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_B': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_B': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_B': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_B': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_B': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'B', 'K'],
)
@triton.jit
def ell_spmm_kernel(
    vals, cols, x, y,
    M: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    stride_vals_m: tl.constexpr,
    stride_vals_k: tl.constexpr,
    stride_cols_m: tl.constexpr,
    stride_cols_k: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_y_m: tl.constexpr,
    stride_y_b: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    bs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)

    mask_y = (rows[:, None] < M) & (bs[None, :] < B)
    acc = tl.zeros((BLOCK_M, BLOCK_B), tl.float32)

    for kk in range(K):
        a_vals = tl.load(
            vals + rows * stride_vals_m + kk * stride_vals_k,
            mask=rows < M,
            other=0.0,
        )

        a_cols = tl.load(
            cols + rows * stride_cols_m + kk * stride_cols_k,
            mask=rows < M,
            other=0,
        )

        x_vals = tl.load(
            x + a_cols[:, None] * stride_x_n + bs[None, :] * stride_x_b,
            mask=(rows[:, None] < M) & (bs[None, :] < B),
            other=0.0,
        )

        acc += a_vals[:, None] * x_vals

    tl.store(
        y + rows[:, None] * stride_y_m + bs[None, :] * stride_y_b,
        acc,
        mask=mask_y,
    )


def ell_spmm(vals, cols, x):
    """
    vals: [M, K]
    cols: [M, K]
    x:    [N, B]
    returns y: [M, B]

    Block sizes are chosen automatically by @triton.autotune.
    """
    M, K = vals.shape
    N, B = x.shape
    y = torch.empty((M, B), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(B, meta['BLOCK_B']),
    )

    ell_spmm_kernel[grid](
        vals, cols, x, y,
        M, B, K,
        vals.stride(0), vals.stride(1),
        cols.stride(0), cols.stride(1),
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
    )

    return y


def print_best_config():
    """Print the best config found by the autotuner, if any."""
    try:
        cfg = ell_spmm_kernel.best_config
        if cfg is not None:
            print(f"Best config: BLOCK_M={cfg.kwargs['BLOCK_M']}, "
                  f"BLOCK_B={cfg.kwargs['BLOCK_B']}, "
                  f"num_warps={cfg.num_warps}, num_stages={cfg.num_stages}")
        else:
            print("No best config yet — run the kernel first.")
    except Exception:
        print("No best config yet — run the kernel first.")


def csr_to_ell(csr: torch.Tensor, K: int | None = None):
    """Convert a PyTorch sparse CSR matrix to ELL format.

    Returns vals: [M, K], cols: [M, K].
    Padded entries use col=0, val=0.
    """
    assert csr.layout == torch.sparse_csr, "Input must be a sparse CSR tensor"

    crow = csr.crow_indices()
    col = csr.col_indices()
    val = csr.values()

    M = csr.shape[0]
    nnz_per_row = crow[1:] - crow[:-1]

    if K is None:
        K = int(nnz_per_row.max().item())

    if torch.any(nnz_per_row > K):
        raise ValueError("Some rows have more nonzeros than K")

    ell_cols = torch.zeros((M, K), device=col.device, dtype=col.dtype)
    ell_vals = torch.zeros((M, K), device=val.device, dtype=val.dtype)

    for k in range(K):
        mask = nnz_per_row > k
        rows = torch.nonzero(mask, as_tuple=False).flatten()
        src_idx = crow[rows] + k
        ell_cols[rows, k] = col[src_idx]
        ell_vals[rows, k] = val[src_idx]

    return ell_vals, ell_cols

