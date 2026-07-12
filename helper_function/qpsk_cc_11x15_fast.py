"""Low-complexity QPSK 11x15 cyclic-cumulant construction.

This module implements the fixed QPSK 11x15 CC feature used by CAPNet without
the general set-partition runtime path in ``pre_cc_matrix_core_from_old``.
It is torch-native and batch-capable, so the full 32768-point IQ tensors can
stay on the selected device.
"""

from __future__ import annotations

from typing import Iterable

import torch


K_LIST = tuple(range(-5, 6))
K_TO_INDEX = {k: i for i, k in enumerate(K_LIST)}
FIRST15_NM = (
    (2, 0),
    (2, 1),
    (2, 2),
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
    (6, 0),
    (6, 1),
    (6, 2),
    (6, 3),
    (6, 4),
    (6, 5),
    (6, 6),
)
NM_TO_COL = {nm: i for i, nm in enumerate(FIRST15_NM)}

ACTIVE_K = {
    (2, 1): (-1, 0, 1),
    (4, 0): (-2, -1, 0, 1, 2),
    (4, 2): (-2, -1, 0, 1, 2),
    (4, 4): (-2, -1, 0, 1, 2),
    (6, 1): (-2, -1, 0, 1, 2),
    (6, 3): (-2, -1, 0, 1, 2),
    (6, 5): (-2, -1, 0, 1, 2),
}


def _as_batch_signal(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    x = torch.as_tensor(x)
    if x.ndim == 1:
        return x.unsqueeze(0), True
    if x.ndim != 2:
        raise ValueError(f"x must have shape (N,) or (B,N), got {tuple(x.shape)}")
    return x, False


def _as_batch_vector(value, batch: int, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(value, device=device, dtype=torch.float64)
    if value.ndim == 0:
        value = value.reshape(1).expand(batch)
    else:
        value = value.reshape(-1)
        if int(value.numel()) == 1:
            value = value.expand(batch)
    if int(value.numel()) != int(batch):
        raise ValueError(f"Expected {batch} values, got {value.numel()}")
    return value


def _empty_sequence(batch: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((batch, len(K_LIST)), device=device, dtype=torch.complex128)


def _assign_k(seq: torch.Tensor, k: int, value: torch.Tensor) -> None:
    seq[:, K_TO_INDEX[int(k)]] = value


def _seq_at(seq: torch.Tensor, k: int) -> torch.Tensor:
    return seq[:, K_TO_INDEX[int(k)]]


def _mask_sequence(seq: torch.Tensor, nm: tuple[int, int]) -> torch.Tensor:
    out = torch.zeros_like(seq)
    for k in ACTIVE_K[nm]:
        out[:, K_TO_INDEX[int(k)]] = seq[:, K_TO_INDEX[int(k)]]
    return out


def _conv2_same_k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(a)
    for ka in K_LIST:
        av = _seq_at(a, ka)
        for kb in K_LIST:
            kout = ka + kb
            if kout in K_TO_INDEX:
                out[:, K_TO_INDEX[kout]] += av * _seq_at(b, kb)
    return out


def _conv3_same_k(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(a)
    for ka in K_LIST:
        av = _seq_at(a, ka)
        for kb in K_LIST:
            abv = av * _seq_at(b, kb)
            for kc in K_LIST:
                kout = ka + kb + kc
                if kout in K_TO_INDEX:
                    out[:, K_TO_INDEX[kout]] += abv * _seq_at(c, kc)
    return out


def _project_moment(
    u: torch.Tensor,
    t0: torch.Tensor,
    f0: torch.Tensor,
    n: int,
    m: int,
    k_values: Iterable[int],
) -> torch.Tensor:
    k_values = tuple(int(k) for k in k_values)
    if not k_values:
        return torch.empty((u.shape[0], 0), device=u.device, dtype=torch.complex128)

    n_samples = int(u.shape[-1])
    k = torch.as_tensor(k_values, device=u.device, dtype=torch.float64)
    alpha = float(n - 2 * m) * f0[:, None] + k[None, :] / t0[:, None]
    t = torch.arange(n_samples, device=u.device, dtype=torch.float64)
    phase = -2.0 * torch.pi * alpha[:, :, None] * t[None, None, :]
    expo = torch.exp((1j * phase).to(torch.complex128))
    return torch.sum(expo * u[:, None, :].to(torch.complex128), dim=-1) / float(n_samples)


def _build_moment_sequences(
    x: torch.Tensor,
    t0,
    f0,
    remove_mean: bool = False,
    quantize_complex64: bool = True,
) -> dict[tuple[int, int], torch.Tensor]:
    x, _ = _as_batch_signal(x)
    batch = int(x.shape[0])
    device = x.device

    if quantize_complex64:
        x = x.to(torch.complex64)
    x = x.to(torch.complex128)
    if remove_mean:
        x = x - x.mean(dim=-1, keepdim=True)

    t0 = _as_batch_vector(t0, batch, device)
    f0 = _as_batch_vector(f0, batch, device)

    x_conj = torch.conj(x)
    abs2 = (x * x_conj).real
    x2 = x * x
    x4 = x2 * x2
    x5 = x4 * x

    u21 = abs2.to(torch.complex128)
    u40 = x4
    u42 = (abs2 * abs2).to(torch.complex128)
    u61 = x5 * x_conj
    u63 = (abs2 * abs2 * abs2).to(torch.complex128)

    moments: dict[tuple[int, int], torch.Tensor] = {}

    m21 = _empty_sequence(batch, device)
    vals = _project_moment(u21, t0, f0, 2, 1, (0, 1))
    _assign_k(m21, 0, vals[:, 0])
    _assign_k(m21, 1, vals[:, 1])
    _assign_k(m21, -1, torch.conj(vals[:, 1]))
    moments[(2, 1)] = m21

    m40 = _empty_sequence(batch, device)
    vals = _project_moment(u40, t0, f0, 4, 0, (-2, -1, 0, 1, 2))
    for i, k in enumerate((-2, -1, 0, 1, 2)):
        _assign_k(m40, k, vals[:, i])
    moments[(4, 0)] = m40

    m42 = _empty_sequence(batch, device)
    vals = _project_moment(u42, t0, f0, 4, 2, (0, 1, 2))
    _assign_k(m42, 0, vals[:, 0])
    _assign_k(m42, 1, vals[:, 1])
    _assign_k(m42, 2, vals[:, 2])
    _assign_k(m42, -1, torch.conj(vals[:, 1]))
    _assign_k(m42, -2, torch.conj(vals[:, 2]))
    moments[(4, 2)] = m42

    m44 = _empty_sequence(batch, device)
    for k in (-2, -1, 0, 1, 2):
        _assign_k(m44, k, torch.conj(_seq_at(m40, -k)))
    moments[(4, 4)] = m44

    m61 = _empty_sequence(batch, device)
    vals = _project_moment(u61, t0, f0, 6, 1, (-2, -1, 0, 1, 2))
    for i, k in enumerate((-2, -1, 0, 1, 2)):
        _assign_k(m61, k, vals[:, i])
    moments[(6, 1)] = m61

    m63 = _empty_sequence(batch, device)
    vals = _project_moment(u63, t0, f0, 6, 3, (0, 1, 2))
    _assign_k(m63, 0, vals[:, 0])
    _assign_k(m63, 1, vals[:, 1])
    _assign_k(m63, 2, vals[:, 2])
    _assign_k(m63, -1, torch.conj(vals[:, 1]))
    _assign_k(m63, -2, torch.conj(vals[:, 2]))
    moments[(6, 3)] = m63

    m65 = _empty_sequence(batch, device)
    for k in (-2, -1, 0, 1, 2):
        _assign_k(m65, k, torch.conj(_seq_at(m61, -k)))
    moments[(6, 5)] = m65

    return moments


def build_qpsk_cc_matrix_11x15_torch(
    x: torch.Tensor,
    t0,
    f0,
    remove_mean: bool = False,
    quantize_complex64: bool = True,
) -> torch.Tensor:
    """Return the QPSK-masked complex CC matrix with shape ``(..., 11, 15)``."""
    x_batch, squeeze = _as_batch_signal(x)
    moments = _build_moment_sequences(
        x_batch,
        t0,
        f0,
        remove_mean=remove_mean,
        quantize_complex64=quantize_complex64,
    )
    batch = int(x_batch.shape[0])
    mat = torch.zeros(
        (batch, len(K_LIST), len(FIRST15_NM)),
        device=x_batch.device,
        dtype=torch.complex128,
    )

    m21 = moments[(2, 1)]
    m40 = moments[(4, 0)]
    m42 = moments[(4, 2)]
    m44 = moments[(4, 4)]
    m61 = moments[(6, 1)]
    m63 = moments[(6, 3)]
    m65 = moments[(6, 5)]

    c21 = m21
    c40 = m40
    c42 = _mask_sequence(m42 - 2.0 * _conv2_same_k(m21, m21), (4, 2))
    c44 = m44
    c61 = _mask_sequence(m61 - 5.0 * _conv2_same_k(m40, m21), (6, 1))
    c63 = _mask_sequence(
        m63 - 9.0 * _conv2_same_k(m42, m21) + 12.0 * _conv3_same_k(m21, m21, m21),
        (6, 3),
    )
    c65 = _mask_sequence(m65 - 5.0 * _conv2_same_k(m44, m21), (6, 5))

    for nm, seq in (
        ((2, 1), c21),
        ((4, 0), c40),
        ((4, 2), c42),
        ((4, 4), c44),
        ((6, 1), c61),
        ((6, 3), c63),
        ((6, 5), c65),
    ):
        mat[:, :, NM_TO_COL[nm]] = _mask_sequence(seq, nm)

    return mat[0] if squeeze else mat


def build_qpsk_cc_abs_11x15_torch(
    x: torch.Tensor,
    t0,
    f0,
    remove_mean: bool = False,
    quantize_complex64: bool = True,
) -> torch.Tensor:
    """Return the post-warped absolute QPSK 11x15 CC matrix."""
    mat = build_qpsk_cc_matrix_11x15_torch(
        x,
        t0,
        f0,
        remove_mean=remove_mean,
        quantize_complex64=quantize_complex64,
    )
    abs_mat = torch.abs(mat).to(torch.float64)
    for col, (n, _m) in enumerate(FIRST15_NM):
        if n in (2, 4, 6):
            abs_mat[..., :, col] = abs_mat[..., :, col] ** (2.0 / float(n))
    return abs_mat


def build_qpsk_cap_input_11x15_torch(
    x: torch.Tensor,
    t0,
    f0,
    power,
    eps: float = 1e-12,
    remove_mean: bool = False,
    quantize_complex64: bool = True,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return CAPNet input ``11x15`` values: warped ``abs(CC) / power``."""
    x_batch, squeeze = _as_batch_signal(x)
    abs_mat = build_qpsk_cc_abs_11x15_torch(
        x_batch,
        t0,
        f0,
        remove_mean=remove_mean,
        quantize_complex64=quantize_complex64,
    )
    power = _as_batch_vector(power, int(x_batch.shape[0]), x_batch.device)
    valid = torch.isfinite(power) & (power > float(eps))
    if not bool(torch.all(valid).detach().cpu().item()):
        raise ValueError("Invalid non-positive or non-finite CC normalization power")
    out = abs_mat / power[:, None, None]
    out = out.to(output_dtype)
    return out[0] if squeeze else out


__all__ = [
    "ACTIVE_K",
    "FIRST15_NM",
    "K_LIST",
    "build_qpsk_cap_input_11x15_torch",
    "build_qpsk_cc_abs_11x15_torch",
    "build_qpsk_cc_matrix_11x15_torch",
]
