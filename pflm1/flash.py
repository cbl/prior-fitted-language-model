from __future__ import annotations

import functools
import importlib
import inspect
import os
from typing import Callable

import torch
import torch.nn.functional as F

_HOPPER = (9, 0)


def _accepts_window(fn: Callable) -> bool:
    try:
        return "window_size" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


@functools.cache
def _load_fa4() -> Callable | None:
    # FlashAttention-4's CuTeDSL kernel. The documented import is
    # `flash_attn.cute`; interim builds nested it under
    # `flash_attn.cute.interface`, so try both.
    for name in ("flash_attn.cute", "flash_attn.cute.interface"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        fn = getattr(module, "flash_attn_func", None)
        if fn is not None and _accepts_window(fn):
            return fn
    return None


@functools.cache
def _load_fa3() -> Callable | None:
    try:
        from flash_attn_interface import flash_attn_func
    except ImportError:
        return None
    return flash_attn_func if _accepts_window(flash_attn_func) else None


@functools.cache
def _load_fa2() -> Callable | None:
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        return None
    return flash_attn_func if _accepts_window(flash_attn_func) else None


def _is_blackwell(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device)[0] >= 10


def _is_hopper(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) == _HOPPER


def resolve_backend(device: torch.device) -> str:
    """Name of the backend that ``sliding_attention`` will dispatch to."""
    pinned = os.environ.get("PFLM1_ATTN_BACKEND")
    if pinned:
        return pinned
    if device.type != "cuda":
        return "sdpa"
    if _is_blackwell(device) and _load_fa4() is not None:
        return "fa4"
    if _is_hopper(device) and _load_fa3() is not None:
        return "fa3"
    if _load_fa2() is not None:
        return "fa2"
    return "sdpa"


_LOADERS = {"fa4": _load_fa4, "fa3": _load_fa3, "fa2": _load_fa2}


def sliding_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window: int,
    scale: float,
) -> torch.Tensor:
    """Causal sliding-window attention. RoPE is already applied.

    ``q`` [B, Sq, Hq, D], ``k``/``v`` [B, Sk, Hkv, D], seq-major. Query ``i``
    attends keys ``j`` with ``0 <= (i + Sk - Sq) - j < window``. Returns
    [B, Sq, Hq, D]. GQA is passed to the kernel natively.
    """
    backend = resolve_backend(q.device)
    if q.dtype == torch.float32:
        backend = "sdpa"  # the flash kernels reject fp32; sdpa is the fp32 path
    if backend != "sdpa":
        fn = _LOADERS[backend]()
        if fn is None:
            raise RuntimeError(f"attention backend {backend!r} is unavailable")
        out = fn(q, k, v, softmax_scale=scale, causal=True, window_size=(window - 1, 0))
        return out[0] if isinstance(out, tuple) else out
    return _sdpa_sliding_attention(q, k, v, window=window, scale=scale)


def sliding_window_mask(
    n_cached: int, seq_len: int, window: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Additive [1, 1, Sq, Sk] mask for the local frame.

    Query ``j`` sits at local position ``n_cached + j`` and attends key ``p``
    iff ``0 <= (n_cached + j) - p < window``. The diagonal is always allowed,
    so no row is fully masked.
    """
    qpos = torch.arange(seq_len, device=device) + n_cached
    kpos = torch.arange(n_cached + seq_len, device=device)
    offset = qpos[:, None] - kpos[None, :]
    allowed = (offset >= 0) & (offset < window)
    mask = torch.zeros(seq_len, n_cached + seq_len, dtype=dtype, device=device)
    return mask.masked_fill_(~allowed, torch.finfo(dtype).min)[None, None]


def _sdpa_sliding_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window: int,
    scale: float,
) -> torch.Tensor:
    """Reference path. Materializes an [Sq, Sk] mask, so it is only tractable
    because the carried window bounds ``Sk`` to ``window - 1 + Sq``."""
    sq, sk = q.shape[1], k.shape[1]
    groups = q.shape[2] // k.shape[2]
    if groups > 1:
        k = k.repeat_interleave(groups, dim=2)
        v = v.repeat_interleave(groups, dim=2)
    mask = sliding_window_mask(sk - sq, sq, window, q.device, q.dtype)
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask, scale=scale,
    ).transpose(1, 2)
