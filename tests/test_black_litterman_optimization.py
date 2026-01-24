"""
Tests for Black-Litterman Portfolio Optimization Module.

QUANT-NEURAL v2026.1 — Task 7.3.1 / 7.3.1.1 / 7.3.1.2

Tests are designed to work whether cvxpy/tensorflow is installed or not.
All functions are fail-safe and should never raise exceptions.
All optimizer outputs must satisfy constraints.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.black_litterman_optimization import (
    black_litterman_posterior,
    calibrate_sector_views,
    compute_dynamic_ic,
    enforce_monthly_units,
    mc_dropout_sector_views,
    optimize_portfolio_cvxpy,
    rmt_denoise_covariance,
)


# =============================================================================
# Test: enforce_monthly_units
# =============================================================================
class TestEnforceMonthlyUnits:
    def test_annualized_to_monthly_conversion(self):
        mu = np.array([0.12, 0.08])
        Sigma = np.array([[0.04, 0.01], [0.01, 0.02]])
        vol = np.array([0.20, 0.14])

        mu_m, Sigma_m, vol_m = enforce_monthly_units(
            mu=mu, Sigma=Sigma, vol=vol, inputs_are_annualized=True
        )

        np.testing.assert_allclose(mu_m, mu / 12, rtol=1e-10)
        np.testing.assert_allclose(Sigma_m, Sigma / 12, rtol=1e-10)
        np.testing.assert_allclose(vol_m, vol / np.sqrt(12), rtol=1e-10)

    def test_already_monthly_no_change(self):
        mu = np.array([0.01, 0.008])
        Sigma = np.array([[0.003, 0.001], [0.001, 0.002]])
        vol = np.array([0.055, 0.04])

        mu_m, Sigma_m, vol_m = enforce_monthly_units(
            mu=mu, Sigma=Sigma, vol=vol, inputs_are_annualized=False
        )

        np.testing.assert_allclose(mu_m, mu, rtol=1e-10)
        np.testing.assert_allclose(Sigma_m, Sigma, rtol=1e-10)
        np.testing.assert_allclose(vol_m, vol, rtol=1e-10)

    def test_vol_none_handled(self):
        mu = np.array([0.12])
        Sigma = np.array([[0.04]])

        mu_m, Sigma_m, vol_m = enforce_monthly_units(
            mu=mu, Sigma=Sigma, vol=None, inputs_are_annualized=True
        )

        assert vol_m is None
        np.testing.assert_allclose(mu_m, mu / 12, rtol=1e-10)

    def test_nan_in_mu_returns_safe_defaults_with_warning(self, caplog):
        mu = np.array([0.12, np.nan])
        Sigma = np.array([[0.04, 0.01], [0.01, 0.02]])

        with caplog.at_level(logging.WARNING):
            mu_m, Sigma_m, vol_m = enforce_monthly_units(
                mu=mu, Sigma=Sigma, vol=None, inputs_are_annualized=True
            )

        assert mu_m.shape == (2,)
        assert Sigma_m.shape == (2, 2)
        assert np.all(np.isfinite(mu_m))
        assert np.all(np.isfinite(Sigma_m))

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("BL_FAILSAFE:enforce_monthly_units" in msg for msg in warning_msgs)

    def test_inf_in_sigma_returns_safe_defaults_with_warning(self, caplog):
        mu = np.array([0.12, 0.08])
        Sigma = np.array([[0.04, np.inf], [0.01, 0.02]])

        with caplog.at_level(logging.WARNING):
            mu_m, Sigma_m, vol_m = enforce_monthly_units(
                mu=mu, Sigma=Sigma, vol=None, inputs_are_annualized=True
            )

        assert mu_m.shape == (2,)
        assert Sigma_m.shape == (2, 2)
        assert np.all(np.isfinite(mu_m))
        assert np.all(np.isfinite(Sigma_m))

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("BL_FAILSAFE:enforce_monthly_units" in msg for msg in warning_msgs)

    def test_shape_mismatch_returns_safe_defaults_with_warning(self, caplog):
        mu = np.array([0.12, 0.08, 0.05])
        Sigma = np.array([[0.04, 0.01], [0.01, 0.02]])

        with caplog.at_level(logging.WARNING):
            mu_m, Sigma_m, vol_m = enforce_monthly_units(
                mu=mu, Sigma=Sigma, vol=None, inputs_are_annualized=True
            )

        assert mu_m.shape == (3,)
        assert Sigma_m.shape == (3, 3)
        assert np.all(np.isfinite(mu_m))
        assert np.all(np.isfinite(Sigma_m))

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("BL_FAILSAFE:enforce_monthly_units" in msg for msg in warning_msgs)


# =============================================================================
# Test: compute_dynamic_ic
# =============================================================================
class TestComputeDynamicIC:
    def test_prob_up_0_5_returns_ic_min(self):
        ic = compute_dynamic_ic(ic_base=0.10, prob_up=0.5, ic_min=0.05)
        assert ic == 0.05

    def test_prob_up_1_returns_full_ic(self):
        ic = compute_dynamic_ic(ic_base=0.10, prob_up=1.0, ic_min=0.05)
        assert ic == 0.10

    def test_prob_up_0_returns_full_ic(self):
        ic = compute_dynamic_ic(ic_base=0.10, prob_up=0.0, ic_min=0.05)
        assert ic == 0.10

    def test_prob_up_clamped_above_1(self):
        ic = compute_dynamic_ic(ic_base=0.10, prob_up=1.5, ic_min=0.05)
        assert ic == 0.10

    def test_prob_up_clamped_below_0(self):
        ic = compute_dynamic_ic(ic_base=0.10, prob_up=-0.5, ic_min=0.05)
        assert ic == 0.10

    def test_ic_min_floor(self):
        ic = compute_dynamic_ic(ic_base=0.02, prob_up=0.5, ic_min=0.05)
        assert ic == 0.05

    def test_nan_input_returns_ic_min_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            ic = compute_dynamic_ic(ic_base=np.nan, prob_up=0.8, ic_min=0.05)

        assert ic == 0.05
        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("BL_FAILSAFE:compute_dynamic_ic" in msg for msg in warning_msgs)


# =============================================================================
# Test: calibrate_sector_views [NEW: shape-lock test]
# =============================================================================
class TestCalibrateSectorViews:
    def test_output_shape(self):
        raw_scores = np.array([0.1, 0.2, 0.15])
        sector_vol = np.array([0.05, 0.06, 0.055])

        Q = calibrate_sector_views(
            raw_scores=raw_scores,
            sector_vol_monthly=sector_vol,
            ic_base=0.10,
            prob_up=0.8,
        )

        assert Q.shape == (3,)

    def test_output_is_finite(self):
        raw_scores = np.array([0.5, 0.3, 0.8, 0.2])
        sector_vol = np.array([0.04, 0.05, 0.06, 0.045])

        Q = calibrate_sector_views(
            raw_scores=raw_scores,
            sector_vol_monthly=sector_vol,
            ic_base=0.10,
            prob_up=0.7,
        )

        assert np.all(np.isfinite(Q))

    def test_zero_std_stability(self):
        raw_scores = np.array([0.5, 0.5, 0.5])
        sector_vol = np.array([0.04, 0.05, 0.06])

        Q = calibrate_sector_views(
            raw_scores=raw_scores,
            sector_vol_monthly=sector_vol,
            ic_base=0.10,
            prob_up=0.7,
        )

        assert np.all(np.isfinite(Q))
        np.testing.assert_allclose(Q, np.zeros(3), atol=1e-10)

    def test_shape_mismatch_returns_K_zeros_with_warning(self, caplog):
        """Shape-lock test: output must ALWAYS be (K,) where K = len(sector_vol_monthly)."""
        K = 10
        sector_vol = np.ones(K) * 0.05

        # Test with shorter raw_scores (K-3)
        raw_scores_short = np.ones(K - 3) * 0.5

        with caplog.at_level(logging.WARNING):
            Q = calibrate_sector_views(
                raw_scores=raw_scores_short,
                sector_vol_monthly=sector_vol,
                ic_base=0.10,
                prob_up=0.8,
            )

        assert Q.shape == (K,), f"Expected shape ({K},), got {Q.shape}"
        assert np.all(np.isfinite(Q))

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("BL_FAILSAFE:calibrate_sector_views:score_length_mismatch" in msg for msg in warning_msgs)

        caplog.clear()

        # Test with longer raw_scores (K+5)
        raw_scores_long = np.ones(K + 5) * 0.5

        with caplog.at_level(logging.WARNING):
            Q = calibrate_sector_views(
                raw_scores=raw_scores_long,
                sector_vol_monthly=sector_vol,
                ic_base=0.10,
                prob_up=0.8,
            )

        assert Q.shape == (K,), f"Expected shape ({K},), got {Q.shape}"
        assert np.all(np.isfinite(Q))

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("BL_FAILSAFE:calibrate_sector_views:score_length_mismatch" in msg for msg in warning_msgs)


# =============================================================================
# Test: mc_dropout_sector_views
# =============================================================================
class TestMCDropoutSectorViews:
    def test_fallback_without_real_model(self, caplog):
        K = 5
        sector_vol_monthly = np.ones(K, dtype=np.float64)
        model = object()
        X = np.zeros((1, 20))

        with caplog.at_level(logging.WARNING):
            Q_mean, Q_var = mc_dropout_sector_views(
                model=model,
                X=X,
                sector_vol_monthly=sector_vol_monthly,
                ic_base=0.10,
                prob_up=0.70,
                n_passes=10,
                seed=123,
            )

        assert Q_mean.shape == (K,)
        assert Q_var.shape == (K,)
        assert np.all(np.isfinite(Q_mean))
        assert np.all(np.isfinite(Q_var))
        assert np.all(Q_var >= 0)

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("BL_FAILSAFE:mc_dropout_sector_views" in msg for msg in warning_msgs)

    def test_determinism_with_same_seed(self, caplog):
        K = 4
        sector_vol_monthly = np.ones(K, dtype=np.float64) * 0.05
        model = object()
        X = np.random.randn(1, 10)

        Q_mean_1, Q_var_1 = mc_dropout_sector_views(
            model=model, X=X, sector_vol_monthly=sector_vol_monthly,
            ic_base=0.10, prob_up=0.70, n_passes=10, seed=123,
        )

        Q_mean_2, Q_var_2 = mc_dropout_sector_views(
            model=model, X=X, sector_vol_monthly=sector_vol_monthly,
            ic_base=0.10, prob_up=0.70, n_passes=10, seed=123,
        )

        np.testing.assert_array_equal(Q_mean_1, Q_mean_2)
        np.testing.assert_array_equal(Q_var_1, Q_var_2)

    def test_dropout_path_invoked_with_mock_model(self, caplog):
        K = 3
        sector_vol_monthly = np.ones(K, dtype=np.float64) * 0.04

        training_calls = []

        class DummyCallable:
            def __call__(self, X, training=False):
                training_calls.append(training)
                if training:
                    return np.array([0.1, 0.2, 0.3]) + np.random.randn(K) * 0.01
                return np.array([0.1, 0.2, 0.3])

        class DummyModel:
            def __init__(self):
                self.model = DummyCallable()

        dummy_model = DummyModel()
        X = np.zeros((1, 5))

        Q_mean, Q_var = mc_dropout_sector_views(
            model=dummy_model, X=X, sector_vol_monthly=sector_vol_monthly,
            ic_base=0.10, prob_up=0.75, n_passes=5, seed=42,
        )

        assert Q_mean.shape == (K,)
        assert Q_var.shape == (K,)
        assert np.all(np.isfinite(Q_mean))
        assert np.all(np.isfinite(Q_var))
        assert len(training_calls) == 5
        assert all(t is True for t in training_calls)

    def test_none_model_returns_fallback(self, caplog):
        K = 3
        sector_vol_monthly = np.ones(K, dtype=np.float64)

        with caplog.at_level(logging.WARNING):
            Q_mean, Q_var = mc_dropout_sector_views(
                model=None, X=np.zeros((1, 10)), sector_vol_monthly=sector_vol_monthly,
                ic_base=0.10, prob_up=0.70, n_passes=5, seed=42,
            )

        assert Q_mean.shape == (K,)
        assert Q_var.shape == (K,)
        assert np.all(np.isfinite(Q_mean))

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("BL_FAILSAFE:mc_dropout_sector_views" in msg for msg in warning_msgs)


# =============================================================================
# Test: rmt_denoise_covariance [NEW: q-clamping detection test]
# =============================================================================
class TestRMTDenoiseCovariance:
    def test_output_shape(self):
        N = 5
        np.random.seed(42)
        X = np.random.randn(100, N)
        Sigma = np.cov(X.T)

        Sigma_den = rmt_denoise_covariance(Sigma, T=100)

        assert Sigma_den.shape == (N, N)

    def test_output_is_finite(self):
        N = 5
        np.random.seed(42)
        X = np.random.randn(100, N)
        Sigma = np.cov(X.T)

        Sigma_den = rmt_denoise_covariance(Sigma, T=100)

        assert np.all(np.isfinite(Sigma_den))

    def test_output_is_psd(self):
        N = 10
        np.random.seed(42)
        X = np.random.randn(200, N)
        Sigma = np.cov(X.T)

        Sigma_den = rmt_denoise_covariance(Sigma, T=200)

        eigenvalues = np.linalg.eigvalsh(Sigma_den)
        assert np.all(eigenvalues >= -1e-8), f"Min eigenvalue: {eigenvalues.min()}"

    def test_output_is_symmetric(self):
        N = 5
        np.random.seed(42)
        X = np.random.randn(100, N)
        Sigma = np.cov(X.T)

        Sigma_den = rmt_denoise_covariance(Sigma, T=100)

        np.testing.assert_allclose(Sigma_den, Sigma_den.T, rtol=1e-10)

    def test_invalid_T_returns_identity(self, caplog):
        N = 3
        Sigma = np.eye(N)

        with caplog.at_level(logging.WARNING):
            Sigma_den = rmt_denoise_covariance(Sigma, T=-10)

        assert Sigma_den.shape == (N, N)
        assert np.all(np.isfinite(Sigma_den))

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("BL_FAILSAFE:rmt_denoise_covariance" in msg for msg in warning_msgs)

    def test_nan_sigma_returns_identity(self, caplog):
        N = 3
        Sigma = np.array([[1.0, np.nan, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]])

        with caplog.at_level(logging.WARNING):
            Sigma_den = rmt_denoise_covariance(Sigma, T=50)

        assert Sigma_den.shape == (N, N)
        assert np.all(np.isfinite(Sigma_den))

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("BL_FAILSAFE:rmt_denoise_covariance" in msg for msg in warning_msgs)

    def test_q_not_clamped_to_one(self):
        """
        Test that q=T/N is NOT forced >= 1.
        
        With N=30, rho=0.3, the one-factor correlation matrix has:
        - Top eigenvalue ≈ 1 + (N-1)*rho ≈ 9.7
        - Other eigenvalues ≈ 1 - rho ≈ 0.7
        
        With T=1 (q ≈ 0.033), correct lambda_plus = (1 + sqrt(1/0.033))^2 ≈ 37.3
        which is > 9.7, so ALL eigenvalues are in noise set.
        
        If q were incorrectly forced to >= 1, lambda_plus ≈ 4, preserving the top eigenvalue.
        
        The test asserts that off-diagonal correlations are near-zero after denoising.
        """
        N = 30
        rho = 0.3
        
        # One-factor correlation matrix
        Corr = np.full((N, N), rho)
        np.fill_diagonal(Corr, 1.0)
        
        # Use as covariance (correlation matrix with unit variance)
        Sigma = Corr.copy()
        
        Sigma_den = rmt_denoise_covariance(Sigma, T=1)
        
        # Extract off-diagonal elements
        off_diag_mask = ~np.eye(N, dtype=bool)
        off_diag_values = Sigma_den[off_diag_mask]
        
        # Off-diagonal mean should be very small (near identity)
        off_diag_mean = np.abs(off_diag_values).mean()
        
        assert off_diag_mean < 1e-3, (
            f"Off-diagonal mean {off_diag_mean:.6f} too large. "
            f"This suggests q was incorrectly clamped to >= 1."
        )


# =============================================================================
# Test: black_litterman_posterior
# =============================================================================
class TestBlackLittermanPosterior:
    def test_output_shape(self):
        N = 5
        K = 2
        Sigma = np.eye(N) * 0.01
        Pi = np.ones(N) * 0.005
        P = np.zeros((K, N))
        P[0, 0] = 1
        P[1, 1] = 1
        Q = np.array([0.01, 0.008])
        Omega = np.eye(K) * 0.001
        tau = 0.05

        mu_post = black_litterman_posterior(Sigma, Pi, P, Q, Omega, tau)

        assert mu_post.shape == (N,)

    def test_output_is_finite(self):
        N = 5
        K = 2
        Sigma = np.eye(N) * 0.01
        Pi = np.ones(N) * 0.005
        P = np.zeros((K, N))
        P[0, 0] = 1
        P[1, 1] = 1
        Q = np.array([0.01, 0.008])
        Omega = np.eye(K) * 0.001
        tau = 0.05

        mu_post = black_litterman_posterior(Sigma, Pi, P, Q, Omega, tau)

        assert np.all(np.isfinite(mu_post))

    def test_no_views_returns_prior(self):
        N = 3
        K = 1
        Sigma = np.eye(N) * 0.01
        Pi = np.array([0.005, 0.006, 0.007])
        P = np.zeros((K, N))
        Q = np.array([0.0])
        Omega = np.eye(K) * 1e10
        tau = 0.05

        mu_post = black_litterman_posterior(Sigma, Pi, P, Q, Omega, tau)

        np.testing.assert_allclose(mu_post, Pi, atol=1e-4)

    def test_shape_mismatch_returns_prior(self, caplog):
        N = 3
        K = 2
        Sigma = np.eye(N) * 0.01
        Pi = np.array([0.005, 0.006, 0.007])
        P = np.zeros((K, N + 1))
        Q = np.array([0.01, 0.008])
        Omega = np.eye(K) * 0.001
        tau = 0.05

        with caplog.at_level(logging.WARNING):
            mu_post = black_litterman_posterior(Sigma, Pi, P, Q, Omega, tau)

        assert mu_post.shape == (N,)
        assert np.all(np.isfinite(mu_post))
        np.testing.assert_allclose(mu_post, Pi, rtol=1e-10)

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("BL_FAILSAFE:black_litterman_posterior" in msg for msg in warning_msgs)


# =============================================================================
# Test: optimize_portfolio_cvxpy [STRENGTHENED]
# =============================================================================
class TestOptimizePortfolioCVXPY:
    def test_returns_finite_weights(self):
        N = 10
        np.random.seed(42)
        mu = np.random.randn(N) * 0.01
        Sigma = np.eye(N) * 0.01
        w_prev = np.ones(N) / N
        sector_map = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8, 9]}

        w = optimize_portfolio_cvxpy(
            mu_monthly=mu, Sigma_monthly=Sigma, sector_map=sector_map, w_prev=w_prev,
        )

        assert w.shape == (N,)
        assert np.all(np.isfinite(w))

    def test_weights_satisfy_stock_cap(self):
        N = 5
        mu = np.array([0.02, 0.015, 0.01, 0.008, 0.005])
        Sigma = np.eye(N) * 0.01
        w_prev = np.zeros(N)
        sector_map = {0: [0, 1], 1: [2, 3, 4]}
        max_stock = 0.10

        w = optimize_portfolio_cvxpy(
            mu_monthly=mu, Sigma_monthly=Sigma, sector_map=sector_map, w_prev=w_prev,
            max_stock_weight=max_stock,
        )

        assert np.all(w <= max_stock + 1e-6)

    def test_weights_satisfy_sector_cap(self):
        N = 6
        mu = np.ones(N) * 0.01
        Sigma = np.eye(N) * 0.01
        w_prev = np.zeros(N)
        sector_map = {0: [0, 1, 2], 1: [3, 4, 5]}
        max_sector = 0.40

        w = optimize_portfolio_cvxpy(
            mu_monthly=mu, Sigma_monthly=Sigma, sector_map=sector_map, w_prev=w_prev,
            max_sector_weight=max_sector,
        )

        for indices in sector_map.values():
            assert np.sum(w[indices]) <= max_sector + 1e-6

    def test_weights_non_negative(self):
        N = 5
        np.random.seed(42)
        mu = np.random.randn(N) * 0.01
        Sigma = np.eye(N) * 0.01
        w_prev = np.ones(N) / N
        sector_map = {0: list(range(N))}

        w = optimize_portfolio_cvxpy(
            mu_monthly=mu, Sigma_monthly=Sigma, sector_map=sector_map, w_prev=w_prev,
        )

        assert np.all(w >= -1e-8)

    def test_allow_cash_true_sum_leq_1(self):
        N = 5
        mu = np.ones(N) * 0.01
        Sigma = np.eye(N) * 0.01
        w_prev = np.zeros(N)
        sector_map = {0: list(range(N))}

        w = optimize_portfolio_cvxpy(
            mu_monthly=mu, Sigma_monthly=Sigma, sector_map=sector_map, w_prev=w_prev,
            allow_cash=True,
        )

        assert np.sum(w) <= 1.0 + 1e-6

    def test_infeasible_triggers_fallback_with_warning_and_satisfies_constraints(self, caplog):
        """Strengthened: assert constraints hold under FINAL caps."""
        N = 5
        mu = np.ones(N) * 0.01
        Sigma = np.eye(N) * 0.01
        w_prev = np.zeros(N)
        sector_map = {0: list(range(N))}

        with caplog.at_level(logging.WARNING):
            w = optimize_portfolio_cvxpy(
                mu_monthly=mu, Sigma_monthly=Sigma, sector_map=sector_map, w_prev=w_prev,
                max_stock_weight=0.01, max_sector_weight=0.02,
                allow_cash=False,
            )

        assert w.shape == (N,)
        assert np.all(np.isfinite(w))
        assert np.all(w >= -1e-6)
        assert abs(np.sum(w) - 1.0) < 1e-5, f"sum(w) = {np.sum(w)}"

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_msgs) > 0

        # Parse final_caps
        final_caps_pattern = r"BL_FAILSAFE:optimize_portfolio_cvxpy:final_caps msw=([\d.]+) mxsw=([\d.]+)"
        final_caps_msgs = [msg for msg in warning_msgs if "final_caps" in msg]
        assert len(final_caps_msgs) > 0, "Expected final_caps log line"

        match = re.search(final_caps_pattern, final_caps_msgs[-1])
        assert match, f"Could not parse final_caps: {final_caps_msgs[-1]}"

        final_msw = float(match.group(1))
        final_mxsw = float(match.group(2))

        assert np.all(w <= final_msw + 1e-6), f"Stock cap violated: max(w)={w.max()}, msw={final_msw}"
        for indices in sector_map.values():
            assert np.sum(w[indices]) <= final_mxsw + 1e-6

    def test_works_without_cvxpy_and_satisfies_constraints(self, monkeypatch, caplog):
        """Strengthened: with hard config + allow_cash=False, must return constraint-satisfying w."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cvxpy":
                raise ImportError("No module named 'cvxpy'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        N = 10
        mu = np.ones(N) * 0.01
        Sigma = np.eye(N) * 0.01
        w_prev = np.zeros(N)
        # Initially infeasible: need sum=1 but 10 * 0.02 = 0.2 < 1
        sector_map = {0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]}
        max_stock = 0.02
        max_sector = 0.05

        with caplog.at_level(logging.WARNING):
            w = optimize_portfolio_cvxpy(
                mu_monthly=mu, Sigma_monthly=Sigma, sector_map=sector_map, w_prev=w_prev,
                max_stock_weight=max_stock, max_sector_weight=max_sector,
                allow_cash=False,
            )

        assert w.shape == (N,)
        assert np.all(np.isfinite(w))
        assert np.all(w >= -1e-6)
        assert abs(np.sum(w) - 1.0) < 1e-5, f"sum(w) = {np.sum(w)}"

        warning_messages = [r.message for r in caplog.records]
        assert any("cvxpy" in msg.lower() for msg in warning_messages)

        # Must have relaxed caps
        relax_caps_msgs = [msg for msg in warning_messages if "relax_caps" in msg]
        assert len(relax_caps_msgs) > 0, "Expected relax_caps log when initial caps infeasible"

        # Parse final caps
        final_caps_pattern = r"BL_FAILSAFE:optimize_portfolio_cvxpy:final_caps msw=([\d.]+) mxsw=([\d.]+)"
        final_caps_msgs = [msg for msg in warning_messages if "final_caps" in msg]
        assert len(final_caps_msgs) > 0

        match = re.search(final_caps_pattern, final_caps_msgs[-1])
        assert match

        final_msw = float(match.group(1))
        final_mxsw = float(match.group(2))

        assert np.all(w <= final_msw + 1e-6)
        for indices in sector_map.values():
            assert np.sum(w[indices]) <= final_mxsw + 1e-6

    def test_empty_sector_map(self):
        N = 3
        mu = np.ones(N) * 0.01
        Sigma = np.eye(N) * 0.01
        w_prev = np.zeros(N)
        sector_map = {}

        w = optimize_portfolio_cvxpy(
            mu_monthly=mu, Sigma_monthly=Sigma, sector_map=sector_map, w_prev=w_prev,
        )

        assert w.shape == (N,)
        assert np.all(np.isfinite(w))

    def test_unexpected_exception_returns_fallback(self, caplog):
        N = 3
        mu = np.ones(N) * 0.01
        Sigma = np.eye(N) * 0.01
        w_prev = np.zeros(N)
        sector_map = "invalid"  # type: ignore

        with caplog.at_level(logging.WARNING):
            w = optimize_portfolio_cvxpy(
                mu_monthly=mu, Sigma_monthly=Sigma, sector_map=sector_map, w_prev=w_prev,
            )

        assert w.shape == (N,)
        assert np.all(np.isfinite(w))

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_messages) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
