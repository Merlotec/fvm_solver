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
    key=['M', 'B', 'K', 'HAS_BIAS'],
)
@triton.jit
def ell_spmm_kernel(
    vals, cols, x, y, b,
    M: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
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

    if HAS_BIAS:
        b_vals = tl.load(b + rows, mask=rows < M, other=0.0)
        acc += b_vals[:, None]

    tl.store(
        y + rows[:, None] * stride_y_m + bs[None, :] * stride_y_b,
        acc,
        mask=mask_y,
    )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=4),
    ],
    key=['M', 'K'],
)
@triton.jit
def _ell_spmv_kernel(
    vals, cols, x, y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_vals_m: tl.constexpr,
    stride_vals_k: tl.constexpr,
    stride_cols_m: tl.constexpr,
    stride_cols_k: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_y_m: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """SpMV: ELL-packed A × vector x = vector y.  One block of rows per program."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < M

    # Gather x indices and values for all K nonzeros in this block of rows
    acc = tl.zeros((BLOCK_M,), tl.float32)
    for kk in range(K):
        a_val = tl.load(
            vals + rows * stride_vals_m + kk * stride_vals_k,
            mask=mask, other=0.0,
        )
        a_col = tl.load(
            cols + rows * stride_cols_m + kk * stride_cols_k,
            mask=mask, other=0,
        )
        x_val = tl.load(x + a_col * stride_x_n, mask=mask, other=0.0)
        acc += a_val * x_val

    tl.store(y + rows * stride_y_m, acc, mask=mask)


def ell_spmv(vals, cols, x, b=None):
    """
    vals: [M, K], cols: [M, K]
    x:    [N] or [N, 1]
    b:    [M] or None   (fused bias: y = A @ x + b)
    returns y: [M] or [M, 1] (matches x dimensionality)

    1D kernel: one block per row chunk, no wasted BLOCK_B dimension.
    Autotuned on (M, K).
    """
    M, K = vals.shape

    # Accept 1D or 2D x
    x_1d = x.ndim == 1
    if x_1d:
        x = x.unsqueeze(-1)   # [N] -> [N, 1]

    N, B = x.shape
    y = torch.empty((M, B), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)

    for bcol in range(B):
        _ell_spmv_kernel[grid](
            vals, cols, x[:, bcol], y[:, bcol],
            M, N, K,
            vals.stride(0), vals.stride(1),
            cols.stride(0), cols.stride(1),
            x.stride(0),
            y.stride(0),
        )

    if b is not None:
        y += b.view(M, 1)

    if x_1d:
        y = y.squeeze(-1)     # [M, 1] -> [M]

    return y


def ell_spmm(vals, cols, x, b=None):
    """
    vals: [M, K], cols: [M, K]
    x:    [N, B]
    b:    [M] or None   (fused bias: y = A @ x + b)
    returns y: [M, B]

    Dispatches to ell_spmv when B == 1.
    Block sizes chosen automatically by @triton.autotune.
    """
    M, K = vals.shape
    N, B = x.shape

    if B == 1:
        return ell_spmv(vals, cols, x, b=b)

    HAS_BIAS = b is not None
    y = torch.empty((M, B), device=x.device, dtype=x.dtype)
    b_ptr = b if HAS_BIAS else torch.empty(1, device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(B, meta['BLOCK_B']),
    )

    ell_spmm_kernel[grid](
        vals, cols, x, y, b_ptr,
        M, B, K, HAS_BIAS,
        vals.stride(0), vals.stride(1),
        cols.stride(0), cols.stride(1),
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
    )

    return y


def print_best_config():
    """Print the best config found by the autotuner, if any."""
    try:
        cfg = _ell_spmv_kernel.best_config
        if cfg is not None:
            print(f'Triton config: {cfg}')
        else:
            print("No best config yet — run the kernel first.")
    except Exception:
        print("Error getting best config.")


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

