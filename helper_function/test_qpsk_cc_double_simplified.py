"""Bitwise regression tests for the production double QPSK CC shortcut."""

from __future__ import annotations

import numpy as np

from . import pre_cc_matrix_core_from_old as cc


def _random_complex64_promoted_to_double(seed: int, size: int = 32768) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.standard_normal(size).astype(np.float32)
    imag = rng.standard_normal(size).astype(np.float32)
    return (real + 1j * imag).astype(np.complex64).astype(np.complex128)


def test_simplified_double_is_bitwise_equal_to_original_first15() -> None:
    previous_runtime = dict(cc.CC_RUNTIME_CFG)
    try:
        cc.configure_cc_runtime(USE_CUDA=False, CUDA_COMPLEX_DTYPE="complex128")
        x = _random_complex64_promoted_to_double(seed=20260804)
        t0 = 18.375
        f0 = 0.0012345

        original = cc.build_estimated_cc_matrix_11x24_from_signal_pattern(
            x, t0, f0, cc.qpsk, remove_mean=False, max_k=5
        )["matrix_complex"][:, :15]
        simplified = cc.build_estimated_qpsk_cc_matrix_11x15_simplified_double(
            x, t0, f0, remove_mean=False, max_k=5
        )["matrix_complex"]

        np.testing.assert_array_equal(simplified, original, strict=True)
    finally:
        cc.configure_cc_runtime(**previous_runtime)


def test_simplified_double_cap_input_is_bitwise_equal() -> None:
    previous_runtime = dict(cc.CC_RUNTIME_CFG)
    try:
        cc.configure_cc_runtime(USE_CUDA=False, CUDA_COMPLEX_DTYPE="complex128")
        x = _random_complex64_promoted_to_double(seed=20260805)
        t0 = 7.8125
        f0 = -0.003125
        power = float(np.mean(np.abs(x) ** 2))

        original = cc.build_estimated_cc_matrix_11x24_from_signal_pattern(
            x, t0, f0, cc.qpsk, remove_mean=False, max_k=5
        )["matrix_complex"]
        simplified = cc.build_estimated_qpsk_cc_matrix_11x15_simplified_double(
            x, t0, f0, remove_mean=False, max_k=5
        )["matrix_complex"]

        _, original_abs = cc.apply_cc_postprocessing_with_power(
            original,
            power,
            cc.CC_NM,
            enable_warping=True,
            enable_scaling=False,
            is_theory=False,
        )
        _, simplified_abs = cc.apply_cc_postprocessing_with_power(
            simplified,
            power,
            cc.CC_NM[:15],
            enable_warping=True,
            enable_scaling=False,
            is_theory=False,
        )
        original_cap = (
            np.asarray(original_abs[:, :15], dtype=np.float32) / np.float32(power)
        )
        simplified_cap = (
            np.asarray(simplified_abs, dtype=np.float32) / np.float32(power)
        )

        np.testing.assert_array_equal(simplified_cap, original_cap, strict=True)
    finally:
        cc.configure_cc_runtime(**previous_runtime)
