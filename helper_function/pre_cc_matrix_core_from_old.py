#!/usr/bin/env python3
"""CC matrix helpers extracted from CC_Matrix_generate.ipynb.

This module is intentionally self-contained so generation notebooks do not
execute or depend on CC_Matrix_generate.ipynb at runtime.  The CC helper code
below is mechanically extracted from section 10.5 of that notebook; only the
imports and CUDA runtime config shim are added here.
"""

from __future__ import annotations

import math
import cmath
from functools import lru_cache

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - torch is expected in the notebook env
    torch = None

CC_RUNTIME_CFG = {
    "USE_CUDA": False,
    "CUDA_DEVICE": "cuda:0",
    "CUDA_COMPLEX_DTYPE": "complex128",
}


def configure_cc_runtime(**kwargs):
    """Update CUDA-related runtime options used by the extracted helpers."""
    CC_RUNTIME_CFG.update(kwargs)
    return dict(CC_RUNTIME_CFG)


def _use_cuda():
    return (
        torch is not None
        and torch.cuda.is_available()
        and bool(CC_RUNTIME_CFG.get("USE_CUDA", False))
    )


def _cuda_dev():
    dev = CC_RUNTIME_CFG.get("CUDA_DEVICE", "cuda:0")
    try:
        d = torch.device(dev)
        _ = torch.empty(1, device=d)
        return d
    except Exception:
        return torch.device("cuda:0")


def _cuda_dtype():
    if str(CC_RUNTIME_CFG.get("CUDA_COMPLEX_DTYPE", "complex128")).lower() == "complex128":
        return torch.complex128, torch.float64
    return torch.complex64, torch.float32


# 10.5 TABLE3 + CC MATRIX FUNCTIONS
# ============================================================
import cmath
from functools import lru_cache

CC_NM = ([(2,m) for m in range(3)] + [(4,m) for m in range(5)] + [(6,m) for m in range(7)] + [(8,m) for m in range(9)])
CC_COL_LABELS = [f"C({n},{m})" for n,m in CC_NM]
CC_K_LIST = list(range(-5, 6))

def _cc_norm_uv(points):
    pts = np.asarray(points, dtype=np.complex128)
    pts = pts - pts.mean()
    return pts / np.sqrt(np.mean(np.abs(pts)**2) + 1e-12)

def _cc_partitions_list(seq):
    if len(seq) == 1:
        return [[[seq[0]]]]
    first = seq[0]
    out, seen = [], set()
    for part in _cc_partitions_list(seq[1:]):
        cand = [[first]] + [b[:] for b in part]
        key = tuple(sorted(tuple(sorted(b)) for b in cand))
        if key not in seen:
            seen.add(key); out.append([list(b) for b in key])
        for i in range(len(part)):
            cand = [b[:] for b in part]
            cand[i] = [first] + cand[i]
            key = tuple(sorted(tuple(sorted(b)) for b in cand))
            if key not in seen:
                seen.add(key); out.append([list(b) for b in key])
    return out

@lru_cache(maxsize=None)
def _cc_partition_block_signatures(n_total, n_nonconj):
    sigs = []
    for part in _cc_partitions_list(list(range(n_total))):
        coeff = ((-1) ** (len(part) - 1)) * math.factorial(len(part) - 1)
        blocks = []
        for block in part:
            p = sum(1 for j in block if j < n_nonconj)
            q = len(block) - p
            blocks.append((p, q))
        sigs.append((coeff, tuple(sorted(blocks))))
    return tuple(sigs)

def _cc_moment(points, p, q):
    pts = np.asarray(points, dtype=np.complex128)
    return np.mean((pts ** p) * (np.conj(pts) ** q))

def _cc_symbol_cumulant(points, n, m):
    n_nonconj = n - m
    cache, out = {}, 0.0 + 0.0j
    for coeff, blocks in _cc_partition_block_signatures(n, n_nonconj):
        prod = 1.0 + 0.0j
        for p, q in blocks:
            if (p, q) not in cache:
                cache[(p, q)] = _cc_moment(points, p, q)
            prod *= cache[(p, q)]
        out += coeff * prod
    if abs(out.imag) < 1e-12: out = out.real + 0j
    if abs(out.real) < 1e-12: out = 0.0 + 1j*out.imag
    return out

def ensure_table3_row_nm_dict_exists():
    global table3_row_nm_dict
    if "table3_row_nm_dict" in globals():
        return table3_row_nm_dict
    const_raw = {
        "BPSK": [-1, +1],
        "QPSK": [1, 1j, -1, -1j],
        "8PSK": [cmath.exp(1j*2*math.pi*k/8) for k in range(8)],
        "16QAM": [x + 1j*y for x in [-3,-1,1,3] for y in [-3,-1,1,3]],
        "64QAM": [x + 1j*y for x in [-7,-5,-3,-1,1,3,5,7] for y in [-7,-5,-3,-1,1,3,5,7]],
        "256QAM": [x + 1j*y for x in range(-15,16,2) for y in range(-15,16,2)],
    }
    const = {k: _cc_norm_uv(v) for k, v in const_raw.items()}
    table3_row_nm_dict = {}
    for name, pts in const.items():
        table3_row_nm_dict[name] = {nm: _cc_symbol_cumulant(pts, nm[0], nm[1]).real for nm in CC_NM}
    return table3_row_nm_dict

# Pattern dictionaries from the CC code.
bpsk = {
    (2, 0): [-1, 0, 1],
    (2, 1): [-1, 0, 1],
    (2, 2): [-1, 0, 1],

    (4, 0): [-2, -1, 0, 1, 2],
    (4, 1): [-2, -1, 0, 1, 2],
    (4, 2): [-2, -1, 0, 1, 2],
    (4, 3): [-2, -1, 0, 1, 2],
    (4, 4): [-2, -1, 0, 1, 2],

    (6, 0): [-2, -1, 0, 1, 2],
    (6, 1): [-2, -1, 0, 1, 2],
    (6, 2): [-2, -1, 0, 1, 2],
    (6, 3): [-2, -1, 0, 1, 2],
    (6, 4): [-2, -1, 0, 1, 2],
    (6, 5): [-2, -1, 0, 1, 2],
    (6, 6): [-2, -1, 0, 1, 2],

    (8, 0): [-3, -2, -1, 0, 1, 2, 3],
    (8, 1): [-3, -2, -1, 0, 1, 2, 3],
    (8, 2): [-3, -2, -1, 0, 1, 2, 3],
    (8, 3): [-3, -2, -1, 0, 1, 2, 3],
    (8, 4): [-3, -2, -1, 0, 1, 2, 3],
    (8, 5): [-3, -2, -1, 0, 1, 2, 3],
    (8, 6): [-3, -2, -1, 0, 1, 2, 3],
    (8, 7): [-3, -2, -1, 0, 1, 2, 3],
    (8, 8): [-3, -2, -1, 0, 1, 2, 3],
}

qpsk = {
    (2, 1): [-1, 0, 1],

    (4, 0): [-2, -1, 0, 1, 2],
    (4, 2): [-2, -1, 0, 1, 2],
    (4, 4): [-2, -1, 0, 1, 2],

    (6, 1): [-2, -1, 0, 1, 2],
    (6, 3): [-2, -1, 0, 1, 2],
    (6, 5): [-2, -1, 0, 1, 2],

    (8, 0): [-3, -2, -1, 0, 1, 2, 3],
    (8, 2): [-3, -2, -1, 0, 1, 2, 3],
    (8, 4): [-3, -2, -1, 0, 1, 2, 3],
    (8, 6): [-3, -2, -1, 0, 1, 2, 3],
    (8, 8): [-3, -2, -1, 0, 1, 2, 3],
}

psk8 = {
    (2, 1): [-1, 0, 1],
    (4, 2): [-2, -1, 0, 1, 2],
    (6, 3): [-2, -1, 0, 1, 2],
    (8, 0): [-3, -2, -1, 0, 1, 2, 3],
    (8, 4): [-3, -2, -1, 0, 1, 2, 3],
    (8, 8): [-3, -2, -1, 0, 1, 2, 3],
}

dqpsk = {
    (2, 1): [-2, 0, 2],

    (4, 0): [-3, -1, 1, 3],
    (4, 2): [-2, 0, 2],
    (4, 4): [-3, -1, 1, 3],

    (6, 1): [-3, -1, 1, 3],
    (6, 3): [-4, -2, 0, 2, 4],
    (6, 5): [-3, -1, 1, 3],

    (8, 0): [-4, -2, 0, 2, 4],
    (8, 2): [-5, -3, -1, 1, 3, 5],
    (8, 4): [-4, -2, 0, 2, 4],
    (8, 6): [-5, -3, -1, 1, 3, 5],
    (8, 8): [-4, -2, 0, 2, 4],
}

sqpsk = {
    (2, 0): [-1,  1],
    (2, 1): [0],
    (2, 2): [-1,  1],

    (4, 0): [-2,  0, 2],
    (4, 1): [-1,  1],
    (4, 2): [-2,  0, 2],
    (4, 3): [-1,  1],
    (4, 4): [-2,  0, 2],

    (6, 0): [-1,  1],
    (6, 1): [-2,  0, 2],
    (6, 2): [-1,  1],
    (6, 3): [-2,  0, 2],
    (6, 4): [-1,  1],
    (6, 5): [-2,  0, 2],
    (6, 6): [-1,  1],

    (8, 0): [-2,  0, 2],
    (8, 1): [-3, -1, 1, 3],
    (8, 2): [-2,  0, 2],
    (8, 3): [-3, -1, 1, 3],
    (8, 4): [-2,  0, 2],
    (8, 5): [-3, -1, 1, 3],
    (8, 6): [-2,  0, 2],
    (8, 7): [-3, -1, 1, 3],
    (8, 8): [-2,  0, 2],
}

CC_PATTERN_MAP = {"bpsk":bpsk,"qpsk":qpsk,"8psk":psk8,"dqpsk":dqpsk,"msk":sqpsk,"16qam":qpsk,"64qam":qpsk,"256qam":qpsk}
CC_PDF_MAP = {"bpsk":"BPSK","qpsk":"QPSK","8psk":"8PSK","dqpsk":"QPSK","msk":"QPSK","16qam":"16QAM","64qam":"64QAM","256qam":"256QAM"}

def get_allowed_k_list(pattern_dict, n, m):
    return list(pattern_dict.get((n, m), []))

def get_allowed_k_set(pattern_dict, n, m):
    return set(get_allowed_k_list(pattern_dict, n, m))

def srrc_pulse(t, T0=1.0, beta=0.35):
    t = np.asarray(t, dtype=float)
    p = np.zeros_like(t, dtype=float)
    eps = 1e-12
    idx = np.ones_like(t, dtype=bool)
    idx0 = np.abs(t) < eps
    if np.any(idx0):
        p[idx0] = (1/np.sqrt(T0)) * (1 + beta*(4/np.pi - 1)); idx[idx0] = False
    if beta > 0:
        tsing = T0/(4*beta)
        idxs = np.abs(np.abs(t) - tsing) < 1e-10
        if np.any(idxs):
            p[idxs] = (beta/np.sqrt(2*T0))*((1+2/np.pi)*np.sin(np.pi/(4*beta)) + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
            idx[idxs] = False
    tg = t[idx]
    if tg.size > 0:
        x = tg/T0
        numerator = np.cos((1+beta)*np.pi*x) + np.sin((1-beta)*np.pi*x)/(4*beta*x)
        denominator = 1 - (4*beta*x)**2
        p[idx] = (4*beta/(np.pi*np.sqrt(T0))) * numerator / denominator
    return p

def build_unit_power_srrc(T0=8.0, beta=0.35, span_symbols=12, pulse_sps=128):
    dt = T0 / pulse_sps
    t = np.arange(-span_symbols*T0/2, span_symbols*T0/2 + dt, dt)
    p = srrc_pulse(t, T0=T0, beta=beta)
    energy = np.sum(np.abs(p)**2) * dt
    p = p * np.sqrt(T0 / (energy + 1e-12))
    return t, p, dt, float(np.sum(np.abs(p)**2) * dt / T0)

def pulse_integral_for_order(alpha, n, t, p, dt):
    return np.sum((p**n) * np.exp(-1j*2*np.pi*alpha*t)) * dt

def build_theoretical_cc_matrix_from_table3_pattern(pdf_name, pattern_dict, T0=8.0, f0=0.0, beta=0.35, span_symbols=12, pulse_sps=128, use_cfo_in_pulse_alpha=False):
    ensure_table3_row_nm_dict_exists()
    if pdf_name not in table3_row_nm_dict:
        raise ValueError(f"{pdf_name} not found in table3_row_nm_dict")
    Ca = table3_row_nm_dict[pdf_name]
    t, p, dt, theory_power = build_unit_power_srrc(T0=T0, beta=beta, span_symbols=span_symbols, pulse_sps=pulse_sps)
    cache = {}
    M = np.zeros((len(CC_K_LIST), len(CC_NM)), dtype=np.complex128)
    for j, (n, m) in enumerate(CC_NM):
        if (n, m) not in pattern_dict: continue
        for k in get_allowed_k_list(pattern_dict, n, m):
            if k not in CC_K_LIST: continue
            alpha = ((n - 2*m)*f0 + k/T0) if use_cfo_in_pulse_alpha else (k/T0)
            if (n, alpha) not in cache:
                cache[(n, alpha)] = pulse_integral_for_order(alpha, n, t, p, dt)
            M[CC_K_LIST.index(k), j] = (Ca[(n, m)] / T0) * cache[(n, alpha)]
    return {"matrix_complex": M, "matrix_abs": np.abs(M), "theory_power": theory_power, "k_list": CC_K_LIST, "col_labels": CC_COL_LABELS}

@lru_cache(maxsize=None)
def _set_partitions_tuple_cc(n):
    if n == 0: return ((),)
    prev = _set_partitions_tuple_cc(n-1)
    out = []
    for p in prev:
        out.append([list(b) for b in p] + [[n-1]])
        for i in range(len(p)):
            p2 = [list(b) for b in p]
            p2[i].append(n-1)
            out.append(p2)
    seen, canon = set(), []
    for p in out:
        key = tuple(sorted([tuple(sorted(b)) for b in p], key=lambda x:(len(x),x)))
        if key not in seen:
            seen.add(key); canon.append(key)
    return tuple(canon)

def _set_partitions_cc(n):
    return [[list(block) for block in part] for part in _set_partitions_tuple_cc(n)]

def cyclic_moment_sequence_tau0_pattern(
    x,
    T0,
    f0,
    n,
    m,
    K,
    allowed_k,
    remove_mean=False,
):
    if not _use_cuda():
        x = np.asarray(x, dtype=np.complex128)

        if remove_mean:
            x = x - np.mean(x)

        N = x.size
        t = np.arange(N, dtype=np.float64)
        Rs = 1.0 / T0

        out = np.zeros(2 * K + 1, dtype=np.complex128)
        allowed_k = sorted([k for k in allowed_k if -K <= k <= K])

        if not allowed_k:
            return out

        u = (x ** (n - m)) * (np.conj(x) ** m)
        beta = (n - 2 * m) * f0 + np.array(allowed_k, dtype=np.float64) * Rs

        vals = (np.exp(-1j * 2 * np.pi * beta[:, None] * t[None, :]) @ u) / N

        for kk, vv in zip(allowed_k, vals):
            out[kk + K] = vv

        return out

    dev = _cuda_dev()
    cdtype, rdtype = _cuda_dtype()

    x_np = np.asarray(x, dtype=np.complex64 if cdtype == torch.complex64 else np.complex128)
    x_t = torch.from_numpy(x_np).to(dev, dtype=cdtype)

    if remove_mean:
        x_t = x_t - torch.mean(x_t)

    N = int(x_t.numel())
    out_t = torch.zeros(2 * K + 1, device=dev, dtype=cdtype)

    allowed_k = sorted([int(k) for k in allowed_k if -K <= int(k) <= K])

    if not allowed_k:
        out = out_t.detach().cpu().numpy().astype(np.complex128)
        del x_t, out_t
        torch.cuda.empty_cache()
        return out

    u_t = (x_t ** int(n - m)) * (torch.conj(x_t) ** int(m))

    k_t = torch.tensor(allowed_k, device=dev, dtype=rdtype)
    Rs = 1.0 / float(T0)

    beta_t = float(n - 2 * m) * float(f0) + k_t * float(Rs)
    t_t = torch.arange(N, device=dev, dtype=rdtype)

    E_t = torch.exp(
        -1j * 2.0 * np.pi * beta_t[:, None] * t_t[None, :]
    ).to(cdtype)

    vals_t = (E_t @ u_t) / float(N)

    for idx, kk in enumerate(allowed_k):
        out_t[kk + K] = vals_t[idx]

    out = out_t.detach().cpu().numpy().astype(np.complex128)

    del x_t, u_t, k_t, beta_t, t_t, E_t, vals_t, out_t
    torch.cuda.empty_cache()

    return out

def _conv_k_sequences(seqs):
    out = np.array([1.0+0j], dtype=np.complex128)
    for s in seqs: out = np.convolve(out, s, mode="full")
    return out

def cyclic_cumulant_sequence_tau0_pattern(x, T0, f0, n, m, K, pattern_dict, remove_mean=False):
    if (n, m) not in pattern_dict:
        return np.zeros(2*K+1, dtype=np.complex128)
    is_conj = np.zeros(n, dtype=np.int32); is_conj[(n-m):] = 1
    parts = _set_partitions_cc(n)
    needed = set()
    for pi in parts:
        for blk in pi:
            nb = len(blk); mb = int(np.sum(is_conj[np.array(blk, dtype=int)])); needed.add((nb, mb))
    Mcache = {}
    for nb, mb in needed:
        Mcache[(nb, mb)] = (np.zeros(2*K+1, dtype=np.complex128) if (nb, mb) not in pattern_dict else
                            cyclic_moment_sequence_tau0_pattern(x, T0, f0, nb, mb, K, get_allowed_k_list(pattern_dict, nb, mb), remove_mean))
    full_half = n*K; center = full_half; C_full = np.zeros(2*full_half+1, dtype=np.complex128)
    for pi in parts:
        coef = ((-1)**(len(pi)-1)) * math.factorial(len(pi)-1)
        seqs = []
        for blk in pi:
            nb = len(blk); mb = int(np.sum(is_conj[np.array(blk, dtype=int)])); seqs.append(Mcache[(nb, mb)])
        conv = _conv_k_sequences(seqs); conv_center = len(pi)*K
        C_full[center-conv_center:center+conv_center+1] += coef * conv
    out = C_full[center-K:center+K+1]
    allowed = get_allowed_k_set(pattern_dict, n, m)
    for kk in range(-K, K+1):
        if kk not in allowed: out[kk+K] = 0j
    return out

def build_estimated_cc_matrix_11x24_from_signal_pattern(x, T0, f0, pattern_dict, remove_mean=False, max_k=5):
    M = np.zeros((len(CC_K_LIST), len(CC_NM)), dtype=np.complex128)
    for j, (n, m) in enumerate(CC_NM):
        M[:, j] = cyclic_cumulant_sequence_tau0_pattern(x, T0, f0, n, m, max_k, pattern_dict, remove_mean)
    return {"matrix_complex": M, "matrix_abs": np.abs(M), "k_list": CC_K_LIST, "col_labels": CC_COL_LABELS}

def get_cc_pdf_and_pattern(signal_type, cfg):
    sig = str(signal_type).lower()
    pat_name = str(cfg.get("CC_PATTERN_OVERRIDE", "auto")).lower()
    if pat_name in ["auto", "none", ""]:
        pattern = CC_PATTERN_MAP.get(sig, qpsk)
    else:
        pattern = {"bpsk":bpsk,"qpsk":qpsk,"8psk":psk8,"psk8":psk8,"dqpsk":dqpsk,"msk":sqpsk,"sqpsk":sqpsk}[pat_name]
    pdf = cfg.get("CC_PDF_OVERRIDE", "auto")
    if str(pdf).lower() in ["auto", "none", ""]:
        pdf = CC_PDF_MAP.get(sig, "QPSK")
    return str(pdf), pattern

def get_cc_T0_from_truth(truth, cfg):
    return float(truth["base_symbol_period"] if str(cfg.get("CC_T0_SOURCE", "T0_eff")).lower().startswith("base") else truth["T0_eff"])

def get_cc_cfo_from_result(result, cfg):
    truth_cfo = float(result["truth"]["carrier_offset_label"])
    src = str(cfg.get("CC_CFO_SOURCE", "residual_after_shift")).lower()
    return truth_cfo if src in ["truth", "label", "truth_label"] else float(result.get("truth_guide_cfo", truth_cfo))

def get_cc_scaling_power(result, prepared, stepB, cfg):
    mode = str(result.get("preprocess_mode", "")).lower()
    src = str(cfg.get("CC_POWER_SOURCE", "auto")).lower()
    if src == "auto": src = "original_total" if mode in ["raw", "none", "no_boi", "no_boi_filter"] else "boi_inband_signal"
    if src == "original_total": return signal_rms_power(result["x_original"])
    if src == "processed_total": return signal_rms_power(result["x_processed"])
    if src == "boi_inband_total": return float(prepared["selected_refined"].get("power_raw", prepared["power_report"]["p_total"])) if prepared is not None else signal_rms_power(result["x_original"])
    if src == "boi_inband_signal": return float(prepared["power_report"].get("p_signal_boi_local", prepared["power_report"]["p_signal_recommended"])) if prepared is not None else signal_rms_power(result["x_original"])
    raise ValueError("Bad CC_POWER_SOURCE")

def apply_cc_postprocessing_with_power(
    matrix_complex,
    power_scale,
    nm_list,
    enable_warping=True,
    enable_scaling=True,
    is_theory=False,
    scale_theory=False,
    eps=1e-12,
):
    """
    First-code-compatible CC post-processing.

    Key differences from the previous integrated version:
      1. Warping uses mag ** (2/n), not max(mag, eps) ** (2/n).
         Therefore exact zeros remain exact zeros.

      2. Scaling follows the first-code rule:
         - theory side is not scaled unless scale_theory=True
         - estimate side can be scaled

      3. No epsilon floor is injected into zero CC entries.
    """
    M = np.array(matrix_complex, dtype=np.complex128, copy=True)

    # --------------------------------------------------------
    # 1) Warping: exact first-code behavior
    # --------------------------------------------------------
    if enable_warping:
        for j, (n, m) in enumerate(nm_list):
            mag = np.abs(M[:, j])
            phs = np.angle(M[:, j])

            # Important:
            # Do NOT use np.maximum(mag, eps).
            # 0 ** fractional_power is still 0.
            mag_warp = mag ** (2.0 / float(n))

            M[:, j] = mag_warp * np.exp(1j * phs)

    # --------------------------------------------------------
    # 2) Scaling
    # --------------------------------------------------------
    if enable_scaling and ((not is_theory) or scale_theory):
        P = float(power_scale)

        if not np.isfinite(P) or P <= eps:
            raise ValueError(f"Invalid CC scaling power: {P}")

        if enable_warping:
            M = M / (P + eps)
        else:
            for j, (n, m) in enumerate(nm_list):
                M[:, j] = M[:, j] / ((P + eps) ** (float(n) / 2.0))

    return M, np.abs(M)

def cc_matrix_abs_df(M, digits=6):
    return pd.DataFrame(np.round(np.abs(M), digits), index=CC_K_LIST, columns=CC_COL_LABELS)

def run_cc_matrix_after_ssca(result, prepared, stepB, cfg, display_tables=True):
    truth = result["truth"]
    pdf_name, pattern_dict = get_cc_pdf_and_pattern(truth["signal_type"], cfg)
    T0_cc = get_cc_T0_from_truth(truth, cfg)
    f0_cc = get_cc_cfo_from_result(result, cfg)
    beta_cc = float(truth["excess_bw"])
    power_scale = get_cc_scaling_power(result, prepared, stepB, cfg)
    theory = build_theoretical_cc_matrix_from_table3_pattern(pdf_name, pattern_dict, T0=T0_cc, f0=f0_cc, beta=beta_cc, span_symbols=int(cfg.get("CC_THEORY_SPAN_SYMBOLS",12)), pulse_sps=int(cfg.get("CC_THEORY_PULSE_SPS",128)), use_cfo_in_pulse_alpha=bool(cfg.get("CC_THEORY_USE_CFO_IN_PULSE_ALPHA",False)))
    estimate = build_estimated_cc_matrix_11x24_from_signal_pattern(np.asarray(result["x_processed"], dtype=np.complex128), T0_cc, f0_cc, pattern_dict, remove_mean=bool(cfg.get("CC_REMOVE_MEAN",False)), max_k=int(cfg.get("CC_MAX_K",5)))
    theory_proc, _ = apply_cc_postprocessing_with_power(theory["matrix_complex"], power_scale, CC_NM, bool(cfg.get("ENABLE_WARPING_THEORY",False)), bool(cfg.get("ENABLE_SCALING_THEORY",False)), is_theory=True, scale_theory=bool(cfg.get("CC_SCALE_THEORY",False)), eps=cfg.get("EPS",1e-12))
    estimate_proc, _ = apply_cc_postprocessing_with_power(estimate["matrix_complex"], power_scale, CC_NM, bool(cfg.get("ENABLE_WARPING_ESTIMATE",True)), bool(cfg.get("ENABLE_SCALING_ESTIMATE",True)), is_theory=False, eps=cfg.get("EPS",1e-12))
    theory_df = cc_matrix_abs_df(theory_proc, int(cfg.get("CC_DISPLAY_DIGITS",6)))
    estimate_df = cc_matrix_abs_df(estimate_proc, int(cfg.get("CC_DISPLAY_DIGITS",6)))
    diff_df = pd.DataFrame(np.round(np.abs(np.abs(estimate_proc)-np.abs(theory_proc)), int(cfg.get("CC_DISPLAY_DIGITS",6))), index=CC_K_LIST, columns=CC_COL_LABELS)
    pack = {"pdf_name":pdf_name, "T0_for_cc":T0_cc, "cfo_for_cc":f0_cc, "beta_for_cc":beta_cc, "power_for_scaling":power_scale, "theory_raw":theory["matrix_complex"], "estimate_raw":estimate["matrix_complex"], "theory_proc":theory_proc, "estimate_proc":estimate_proc, "theory_abs_df":theory_df, "estimate_abs_df":estimate_df, "diff_abs_df":diff_df}
    if display_tables:
        print("\n" + "="*80)
        print("CC MATRIX | theory vs Eq.(10)(11) estimate")
        print("="*80)
        print(f"dataset           : {result['dataset_name']}")
        print(f"signal_index      : {result['signal_index']}")
        print(f"signal_type       : {truth['signal_type']}")
        print(f"pdf_name          : {pdf_name}")
        print(f"preprocess_mode   : {result.get('preprocess_mode','')}")
        print(f"boi_algorithm     : {result.get('boi_algorithm','')}")
        print(f"T0_for_cc         : {T0_cc:.12e}")
        print(f"CFO_for_cc        : {f0_cc:+.12e}")
        print(f"CFO_source        : {cfg.get('CC_CFO_SOURCE','residual_after_shift')}")
        print(f"power_for_scaling : {power_scale:.12e}")
        print(f"power_source      : {cfg.get('CC_POWER_SOURCE','auto')}")
        print(f"warping theory    : {cfg.get('ENABLE_WARPING_THEORY',False)}")
        print(f"scaling theory    : {cfg.get('ENABLE_SCALING_THEORY',False)}")
        print(f"warping estimate  : {cfg.get('ENABLE_WARPING_ESTIMATE',True)}")
        print(f"scaling estimate  : {cfg.get('ENABLE_SCALING_ESTIMATE',True)}")
        print("\n===== CC theory | abs matrix ====="); display(theory_df)
        print("\n===== CC estimate | abs matrix ====="); display(estimate_df)
        if cfg.get("DISPLAY_CC_DIFF_TABLE", True):
            print("\n===== abs(|estimate| - |theory|) ====="); display(diff_df)
    return pack

# ============================================================
