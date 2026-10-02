from __future__ import annotations

import torch
import torch.nn.functional as F


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """The kernel's ``use_qk_l2norm_in_kernel`` normalization, in torch."""
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def delta_rule_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated delta rule, ``S_t = (I - beta_t k_t k_t^T) exp(g_t) S_{t-1}
    + beta_t k_t v_t^T`` then ``o_t = S_t^T q_t``, with ``q``/``k``
    l2-normalized here as the fused kernel normalizes them in-kernel.

    ``q``/``k`` [B, T, H, Dk], ``v`` [B, T, H, Dv], ``beta`` [B, T, H],
    ``g`` [B, T, H] holding the per-head log-decay (``<= 0``).
    Returns ``(out [B, T, H, Dv], final_state [B, H, Dk, Dv])``.
    """
    dtype, t = q.dtype, q.shape[1]
    q, k = l2norm(q), l2norm(k)
    g = g.unsqueeze(-1)

    q, k, v, g = (x.transpose(1, 2).float() for x in (q, k, v, g))
    beta = beta.transpose(1, 2).float()
    q = q * q.shape[-1] ** -0.5

    pad = -t % chunk_size
    if pad:
        q, k, v, g = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v, g))
        beta = F.pad(beta, (0, pad))

    b, h, dk, dv = q.shape[0], q.shape[1], q.shape[-1], v.shape[-1]
    chunks = (t + pad) // chunk_size
    shape = (b, h, chunks, chunk_size)
    q, k, v, g = (x.reshape(*shape, x.shape[-1]) for x in (q, k, v, g))
    beta = beta.reshape(*shape)

    # decay[i, j] = exp(gcum_i - gcum_j) for i >= j, else 0. Masking before the
    # exponential keeps the (positive, unbounded) upper triangle from overflowing.
    gcum = g.cumsum(-2)
    causal = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device))
    decay = (gcum.unsqueeze(-2) - gcum.unsqueeze(-3)).masked_fill(
        ~causal[..., None], float("-inf")
    ).exp()

    # WY representation of the delta-rule updates within a chunk: invert the
    # strictly-lower matrix of pairwise key interactions by forward substitution.
    kb = k * beta[..., None]
    attn = -(decay * kb.unsqueeze(-2) * k.unsqueeze(-3)).sum(-1)
    attn = attn.tril(-1)
    for i in range(1, chunk_size):
        # Both operands are cloned: the write below aliases the block we read.
        row, block = attn[..., i, :i].clone(), attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * block).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    w = attn @ (kb * gcum.exp())
    u = attn @ (v * beta[..., None])

    state = (
        torch.zeros(b, h, dk, dv, dtype=torch.float32, device=q.device)
        if initial_state is None
        else initial_state.float()
    )
    out = torch.empty_like(v)
    for c in range(chunks):
        q_c, k_c, gcum_c, decay_c = q[:, :, c], k[:, :, c], gcum[:, :, c], decay[:, :, c]
        intra = (decay_c * q_c.unsqueeze(-2) * k_c.unsqueeze(-3)).sum(-1)
        v_new = u[:, :, c] - w[:, :, c] @ state
        out[:, :, c] = (q_c * gcum_c.exp()) @ state + intra @ v_new
        last = gcum_c[..., -1:, :]
        state = state * last.squeeze(-2).unsqueeze(-1).exp() + (
            k_c * (last - gcum_c).exp()
        ).transpose(-1, -2) @ v_new

    out = out.reshape(b, h, chunks * chunk_size, dv)[:, :, :t]
    return out.transpose(1, 2).to(dtype), state
