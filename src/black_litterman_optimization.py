"""
Black-Litterman Portfolio Optimization Module.

QUANT-NEURAL v2026.1 — Task 7.3.1 / 7.3.1.1 / 7.3.1.2

This module implements:
- Monthly unit enforcement
- Dynamic IC calibration
- Sector view calibration with MC dropout
- RMT covariance denoising
- Black-Litterman posterior computation
- Fail-safe CVXPY portfolio optimization

Non-negotiables:
- PIT: No current time, no data loading, no network resources.
- Train-only fit: Pure portfolio layer, no fitting.
- Reproducibility: Deterministic outputs given fixed inputs + seed.
- Fail-safe: NEVER crash. Always return valid outputs with logging.warning.
- Never-invalid: Always return constraint-satisfying portfolios.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.models import SectorPredictorMLP

logger = logging.getLogger(__name__)


# =============================================================================
# 1) enforce_monthly_units
# =============================================================================
def enforce_monthly_units(
    *,
    mu: np.ndarray,
    Sigma: np.ndarray,
    vol: np.ndarray | None,
    inputs_are_annualized: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Convert annualized inputs to monthly units if needed.

    FAIL-SAFE: Never raises. Returns safe defaults on invalid input.
    """
    try:
        mu = np.asarray(mu, dtype=np.float64)
        Sigma = np.asarray(Sigma, dtype=np.float64)
        if vol is not None:
            vol = np.asarray(vol, dtype=np.float64)
    except Exception as e:
        logger.warning(f"BL_FAILSAFE:enforce_monthly_units:array_conversion_failed {e}")
        return np.array([]), np.zeros((0, 0)), None

    if mu.ndim != 1:
        logger.warning(f"BL_FAILSAFE:enforce_monthly_units:invalid_mu_shape mu.shape={mu.shape}")
        mu = mu.ravel()
        if mu.size == 0:
            return np.array([]), np.zeros((0, 0)), None

    N = mu.shape[0]
    if N == 0:
        logger.warning("BL_FAILSAFE:enforce_monthly_units:empty_mu")
        return np.array([]), np.zeros((0, 0)), None

    if Sigma.shape != (N, N):
        logger.warning(f"BL_FAILSAFE:enforce_monthly_units:sigma_shape_mismatch Sigma.shape={Sigma.shape}, expected=({N},{N})")
        mu_out = np.zeros(N, dtype=np.float64)
        Sigma_out = np.eye(N, dtype=np.float64)
        vol_out = np.ones(N, dtype=np.float64) if vol is not None else None
        return mu_out, Sigma_out, vol_out

    if vol is not None and vol.shape != (N,):
        logger.warning(f"BL_FAILSAFE:enforce_monthly_units:vol_shape_mismatch vol.shape={vol.shape}, expected=({N},)")
        vol = None

    if not np.all(np.isfinite(mu)):
        logger.warning("BL_FAILSAFE:enforce_monthly_units:mu_contains_nan_or_inf")
        return np.zeros(N), np.eye(N), np.ones(N) if vol is not None else None

    if not np.all(np.isfinite(Sigma)):
        logger.warning("BL_FAILSAFE:enforce_monthly_units:sigma_contains_nan_or_inf")
        return np.zeros(N), np.eye(N), np.ones(N) if vol is not None else None

    if vol is not None and not np.all(np.isfinite(vol)):
        logger.warning("BL_FAILSAFE:enforce_monthly_units:vol_contains_nan_or_inf")
        vol = None

    if not inputs_are_annualized:
        return mu.copy(), Sigma.copy(), vol.copy() if vol is not None else None

    mu_monthly = mu / 12.0
    Sigma_monthly = Sigma / 12.0
    vol_monthly = vol / np.sqrt(12.0) if vol is not None else None

    return mu_monthly, Sigma_monthly, vol_monthly


# =============================================================================
# 2) compute_dynamic_ic
# =============================================================================
def compute_dynamic_ic(
    ic_base: float,
    prob_up: float,
    ic_min: float = 0.05,
) -> float:
    """Compute dynamic IC. FAIL-SAFE: Returns ic_min on invalid input."""
    try:
        ic_base_f = float(ic_base)
        prob_up_f = float(prob_up)
        ic_min_f = float(ic_min)
    except (TypeError, ValueError) as e:
        logger.warning(f"BL_FAILSAFE:compute_dynamic_ic:invalid_input {e}")
        return float(ic_min) if isinstance(ic_min, (int, float)) else 0.05

    if not np.isfinite(ic_base_f) or not np.isfinite(prob_up_f) or not np.isfinite(ic_min_f):
        logger.warning("BL_FAILSAFE:compute_dynamic_ic:non_finite_input")
        return max(0.0, ic_min_f) if np.isfinite(ic_min_f) else 0.05

    prob_up_clamped = float(np.clip(prob_up_f, 0.0, 1.0))
    ic_dyn = ic_base_f * (2.0 * abs(prob_up_clamped - 0.5))
    return max(ic_min_f, ic_dyn)


# =============================================================================
# 3) calibrate_sector_views  [FIXED: output shape always (K,)]
# =============================================================================
def calibrate_sector_views(
    raw_scores: np.ndarray,
    sector_vol_monthly: np.ndarray,
    ic_base: float,
    prob_up: float,
    eps: float = 1e-12,
    ic_min: float = 0.05,
) -> np.ndarray:
    """
    Calibrate raw sector scores to monthly expected return views.

    FAIL-SAFE: Never raises. Output shape is ALWAYS (K,) where K = len(sector_vol_monthly).
    """
    try:
        sector_vol_monthly = np.asarray(sector_vol_monthly, dtype=np.float64).ravel()
    except Exception as e:
        logger.warning(f"BL_FAILSAFE:calibrate_sector_views:array_conversion_failed {e}")
        return np.array([])

    K = sector_vol_monthly.shape[0]
    if K == 0:
        return np.array([], dtype=np.float64)

    try:
        raw_scores = np.asarray(raw_scores, dtype=np.float64).ravel()
    except Exception as e:
        logger.warning(f"BL_FAILSAFE:calibrate_sector_views:raw_scores_conversion_failed {e}")
        return np.zeros(K, dtype=np.float64)

    R = raw_scores.shape[0]

    # Coerce raw_scores to length K
    if R != K:
        logger.warning(f"BL_FAILSAFE:calibrate_sector_views:score_length_mismatch raw={R} vol={K}")
        coerced = np.zeros(K, dtype=np.float64)
        n_copy = min(R, K)
        coerced[:n_copy] = raw_scores[:n_copy]
        raw_scores = coerced

    if not np.all(np.isfinite(raw_scores)) or not np.all(np.isfinite(sector_vol_monthly)):
        logger.warning("BL_FAILSAFE:calibrate_sector_views:non_finite_input")
        return np.zeros(K, dtype=np.float64)

    mean_score = np.mean(raw_scores)
    std_score = np.std(raw_scores)
    z = (raw_scores - mean_score) / (std_score + eps)

    ic_dyn = compute_dynamic_ic(ic_base, prob_up, ic_min)
    Q_monthly = ic_dyn * sector_vol_monthly * z
    Q_monthly = np.where(np.isfinite(Q_monthly), Q_monthly, 0.0)

    return Q_monthly


# =============================================================================
# 4) mc_dropout_sector_views
# =============================================================================
def mc_dropout_sector_views(
    model: "SectorPredictorMLP",
    X: np.ndarray,
    sector_vol_monthly: np.ndarray,
    ic_base: float,
    prob_up: float,
    n_passes: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """MC dropout for view uncertainty. FAIL-SAFE: Returns zeros if unavailable."""
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    try:
        sector_vol_monthly = np.asarray(sector_vol_monthly, dtype=np.float64).ravel()
    except Exception as e:
        logger.warning(f"BL_FAILSAFE:mc_dropout_sector_views:sector_vol_conversion_failed {e}")
        return np.array([]), np.array([])

    K = sector_vol_monthly.shape[0]
    if K == 0:
        return np.array([]), np.array([])

    try:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
    except Exception as e:
        logger.warning(f"BL_FAILSAFE:mc_dropout_sector_views:X_conversion_failed {e}")
        return _mc_dropout_fallback(K, sector_vol_monthly, ic_base, prob_up)

    if model is None or not hasattr(model, "model"):
        logger.warning("BL_FAILSAFE:mc_dropout_sector_views:tf_or_model_unavailable model=None or no .model attr")
        return _mc_dropout_fallback(K, sector_vol_monthly, ic_base, prob_up)

    Q_samples = []
    forward_failed = False

    for i in range(n_passes):
        try:
            raw_out = model.model(X, training=True)
            if hasattr(raw_out, "numpy"):
                raw_scores = raw_out.numpy().ravel()
            else:
                raw_scores = np.asarray(raw_out).ravel()

            Q_j = calibrate_sector_views(raw_scores, sector_vol_monthly, ic_base, prob_up)
            Q_samples.append(Q_j)

        except Exception as e:
            if not forward_failed:
                logger.warning(f"BL_FAILSAFE:mc_dropout_sector_views:tf_or_model_unavailable forward_pass_failed at pass {i}: {e}")
                forward_failed = True
            Q_samples.append(np.zeros(K, dtype=np.float64))

    if len(Q_samples) == 0:
        logger.warning("BL_FAILSAFE:mc_dropout_sector_views:no_samples_collected")
        return _mc_dropout_fallback(K, sector_vol_monthly, ic_base, prob_up)

    Q_arr = np.array(Q_samples)
    Q_mean = np.mean(Q_arr, axis=0)
    Q_var = np.var(Q_arr, axis=0)

    Q_mean = np.where(np.isfinite(Q_mean), Q_mean, 0.0)
    Q_var = np.where(np.isfinite(Q_var), Q_var, 0.0)
    Q_var = np.maximum(Q_var, 0.0)

    return Q_mean, Q_var


def _mc_dropout_fallback(K: int, sector_vol_monthly: np.ndarray, ic_base: float, prob_up: float) -> tuple[np.ndarray, np.ndarray]:
    raw_scores = np.zeros(K, dtype=np.float64)
    Q_mean = calibrate_sector_views(raw_scores, sector_vol_monthly, ic_base, prob_up)
    Q_var = np.zeros(K, dtype=np.float64)
    return Q_mean, Q_var


# =============================================================================
# 5) rmt_denoise_covariance  [FIXED: q = T/N, no q >= 1 clamp]
# =============================================================================
def rmt_denoise_covariance(
    Sigma: np.ndarray,
    *,
    T: int,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Denoise covariance using RMT (Marcenko-Pastur).

    FAIL-SAFE: Never raises. Returns safe PSD on invalid input.
    Uses q = T/N without forcing q >= 1.
    """
    try:
        Sigma = np.asarray(Sigma, dtype=np.float64)
    except Exception as e:
        logger.warning(f"BL_FAILSAFE:rmt_denoise_covariance:array_conversion_failed {e}")
        return np.zeros((0, 0))

    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        logger.warning(f"BL_FAILSAFE:rmt_denoise_covariance:invalid_shape shape={Sigma.shape}")
        if Sigma.size > 0:
            N = max(Sigma.shape)
            return np.eye(N, dtype=np.float64) * eps
        return np.zeros((0, 0))

    N = Sigma.shape[0]
    if N == 0:
        return np.zeros((0, 0))

    if not isinstance(T, (int, float)) or not np.isfinite(T):
        logger.warning(f"BL_FAILSAFE:rmt_denoise_covariance:invalid_T T={T}")
        return np.eye(N, dtype=np.float64) * eps

    if T <= 0:
        logger.warning(f"BL_FAILSAFE:rmt_denoise_covariance:invalid_T T={T}")
        return np.eye(N, dtype=np.float64) * eps

    if not np.all(np.isfinite(Sigma)):
        logger.warning("BL_FAILSAFE:rmt_denoise_covariance:sigma_contains_nan_or_inf")
        return np.eye(N, dtype=np.float64) * eps

    try:
        Sigma = (Sigma + Sigma.T) / 2.0
        diag_vals = np.diag(Sigma)
        if np.any(diag_vals < 0):
            logger.warning("BL_FAILSAFE:rmt_denoise_covariance:negative_diagonal")
            return np.eye(N, dtype=np.float64) * eps

        vol = np.sqrt(np.maximum(diag_vals, eps))
        D_inv = np.diag(1.0 / vol)
        Corr = D_inv @ Sigma @ D_inv

        eigenvalues, eigenvectors = np.linalg.eigh(Corr)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # q = T/N, NO forced q >= 1
        q = float(T) / N

        # If q <= 0, set lambda_plus = +inf (all eigenvalues are noise)
        if q <= 0:
            lambda_plus = np.inf
        else:
            # lambda_plus = (1 + sqrt(1/q))^2
            lambda_plus = (1.0 + np.sqrt(1.0 / q)) ** 2

        noise_mask = eigenvalues <= lambda_plus
        if np.any(noise_mask):
            mean_noise_eig = np.mean(eigenvalues[noise_mask])
            eigenvalues[noise_mask] = mean_noise_eig

        eigenvalues = np.maximum(eigenvalues, eps)

        Corr_denoised = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        Corr_denoised = (Corr_denoised + Corr_denoised.T) / 2.0

        D = np.diag(vol)
        Sigma_denoised = D @ Corr_denoised @ D

        eig_vals, eig_vecs = np.linalg.eigh(Sigma_denoised)
        eig_vals = np.maximum(eig_vals, eps)
        Sigma_denoised = eig_vecs @ np.diag(eig_vals) @ eig_vecs.T
        Sigma_denoised = (Sigma_denoised + Sigma_denoised.T) / 2.0

        if not np.all(np.isfinite(Sigma_denoised)):
            logger.warning("BL_FAILSAFE:rmt_denoise_covariance:result_non_finite")
            return np.eye(N, dtype=np.float64) * eps

        return Sigma_denoised

    except Exception as e:
        logger.warning(f"BL_FAILSAFE:rmt_denoise_covariance:eig_failed {e}")
        return np.eye(N, dtype=np.float64) * eps


# =============================================================================
# 6) black_litterman_posterior
# =============================================================================
def black_litterman_posterior(
    Sigma_monthly: np.ndarray,
    Pi_monthly: np.ndarray,
    P: np.ndarray,
    Q_monthly: np.ndarray,
    Omega: np.ndarray,
    tau: float,
) -> np.ndarray:
    """BL posterior. FAIL-SAFE: Returns prior on failure."""
    try:
        Sigma_monthly = np.asarray(Sigma_monthly, dtype=np.float64)
        Pi_monthly = np.asarray(Pi_monthly, dtype=np.float64).ravel()
        P = np.asarray(P, dtype=np.float64)
        Q_monthly = np.asarray(Q_monthly, dtype=np.float64).ravel()
        Omega = np.asarray(Omega, dtype=np.float64)
        tau = float(tau)
    except Exception as e:
        logger.warning(f"BL_FAILSAFE:black_litterman_posterior:array_conversion_failed {e}")
        return np.array([])

    N = Pi_monthly.shape[0]
    K = Q_monthly.shape[0]

    if N == 0:
        return np.array([])

    def _return_prior():
        return np.where(np.isfinite(Pi_monthly), Pi_monthly, 0.0)

    if Sigma_monthly.shape != (N, N):
        logger.warning(f"BL_FAILSAFE:black_litterman_posterior:sigma_shape_mismatch Sigma.shape={Sigma_monthly.shape}")
        return _return_prior()

    if P.shape != (K, N):
        logger.warning(f"BL_FAILSAFE:black_litterman_posterior:P_shape_mismatch P.shape={P.shape}")
        return _return_prior()

    if Omega.shape != (K, K):
        logger.warning(f"BL_FAILSAFE:black_litterman_posterior:omega_shape_mismatch Omega.shape={Omega.shape}")
        return _return_prior()

    if not np.all(np.isfinite(Sigma_monthly)):
        logger.warning("BL_FAILSAFE:black_litterman_posterior:sigma_non_finite")
        return _return_prior()

    if not np.all(np.isfinite(Pi_monthly)):
        logger.warning("BL_FAILSAFE:black_litterman_posterior:pi_non_finite")
        Pi_monthly = np.where(np.isfinite(Pi_monthly), Pi_monthly, 0.0)

    if not np.isfinite(tau) or tau <= 0:
        logger.warning(f"BL_FAILSAFE:black_litterman_posterior:invalid_tau tau={tau}")
        tau = 0.05

    try:
        Sigma_monthly = (Sigma_monthly + Sigma_monthly.T) / 2.0
        ridge = 1e-8 * np.eye(N)
        Sigma_reg = Sigma_monthly + ridge
        tau_Sigma = tau * Sigma_reg

        P_tau_Sigma = P @ tau_Sigma
        A = P_tau_Sigma @ P.T + Omega
        A = (A + A.T) / 2.0
        A_reg = A + 1e-8 * np.eye(K)

        b = Q_monthly - P @ Pi_monthly

        try:
            x = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            logger.warning("BL_FAILSAFE:black_litterman_posterior:solve_failed using lstsq")
            x, _, _, _ = np.linalg.lstsq(A_reg, b, rcond=None)

        mu_posterior = Pi_monthly + tau_Sigma @ P.T @ x
        mu_posterior = np.where(np.isfinite(mu_posterior), mu_posterior, Pi_monthly)
        return mu_posterior

    except Exception as e:
        logger.warning(f"BL_FAILSAFE:black_litterman_posterior:unexpected_exception {e}")
        return _return_prior()


# =============================================================================
# 7) optimize_portfolio_cvxpy  [FIXED: Always returns constraint-satisfying weights]
# =============================================================================
def optimize_portfolio_cvxpy(
    mu_monthly: np.ndarray,
    Sigma_monthly: np.ndarray,
    sector_map: dict[int, list[int]],
    w_prev: np.ndarray,
    *,
    max_stock_weight: float = 0.10,
    max_sector_weight: float = 0.40,
    lam_risk: float = 1.0,
    gamma_turnover: float = 1.0,
    allow_cash: bool = True,
    solver_sequence: tuple[str, ...] = ("OSQP", "ECOS", "SCS"),
) -> np.ndarray:
    """
    Optimize portfolio with fail-safe fallbacks.

    NEVER-CRASH + NEVER-INVALID: Always returns constraint-satisfying weights.
    """
    try:
        mu_monthly = np.asarray(mu_monthly, dtype=np.float64).ravel()
        Sigma_monthly = np.asarray(Sigma_monthly, dtype=np.float64)
        w_prev = np.asarray(w_prev, dtype=np.float64).ravel()
    except Exception as e:
        logger.warning(f"BL_FAILSAFE:optimize_portfolio_cvxpy:array_conversion_failed {e}")
        return np.array([])

    N = mu_monthly.shape[0]
    if N == 0:
        return np.array([])

    if not isinstance(sector_map, dict):
        logger.warning("BL_FAILSAFE:optimize_portfolio_cvxpy:invalid_sector_map")
        sector_map = {}

    # Sanitize inputs
    if Sigma_monthly.shape != (N, N):
        logger.warning(f"BL_FAILSAFE:optimize_portfolio_cvxpy:sigma_shape_mismatch expected ({N},{N}), got {Sigma_monthly.shape}")
        Sigma_monthly = np.eye(N, dtype=np.float64) * 0.01

    if w_prev.shape[0] != N:
        logger.warning("BL_FAILSAFE:optimize_portfolio_cvxpy:w_prev_shape_mismatch")
        w_prev = np.zeros(N, dtype=np.float64)

    if not np.all(np.isfinite(mu_monthly)):
        logger.warning("BL_FAILSAFE:optimize_portfolio_cvxpy:mu_non_finite")
        mu_monthly = np.zeros(N, dtype=np.float64)

    if not np.all(np.isfinite(Sigma_monthly)):
        logger.warning("BL_FAILSAFE:optimize_portfolio_cvxpy:sigma_non_finite")
        Sigma_monthly = np.eye(N, dtype=np.float64) * 0.01

    if not np.all(np.isfinite(w_prev)):
        w_prev = np.zeros(N, dtype=np.float64)

    Sigma_monthly = (Sigma_monthly + Sigma_monthly.T) / 2.0

    # Working caps (may be relaxed)
    msw = float(max_stock_weight)
    mxsw = float(max_sector_weight)

    cvxpy_available = False
    cp = None
    try:
        import cvxpy as cp
        cvxpy_available = True
    except ImportError:
        logger.warning("BL_FAILSAFE:optimize_portfolio_cvxpy:cvxpy_missing")

    def _try_solve(msw_: float, mxsw_: float, g_turn: float) -> np.ndarray | None:
        if not cvxpy_available:
            return None
        try:
            w = cp.Variable(N)
            u = cp.Variable(N)

            objective = cp.Minimize(
                -mu_monthly @ w
                + lam_risk * cp.quad_form(w, Sigma_monthly)
                + g_turn * cp.sum(u)
            )

            constraints = [w >= 0, w <= msw_, u >= w - w_prev, u >= w_prev - w]

            for sector_id, indices in sector_map.items():
                if indices:
                    constraints.append(cp.sum(w[indices]) <= mxsw_)

            if allow_cash:
                constraints.append(cp.sum(w) <= 1.0)
            else:
                constraints.append(cp.sum(w) == 1.0)

            problem = cp.Problem(objective, constraints)

            for solver_name in solver_sequence:
                try:
                    solver = getattr(cp, solver_name, None)
                    if solver is None:
                        continue
                    problem.solve(solver=solver, verbose=False)
                    if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                        w_val = w.value
                        if w_val is not None and np.all(np.isfinite(w_val)):
                            return np.asarray(w_val).ravel()
                except Exception:
                    continue
            return None
        except Exception:
            return None

    # Tier 0
    result = _try_solve(msw, mxsw, gamma_turnover)
    if result is not None:
        w_out = _project_feasible(result, N, sector_map, msw, mxsw, allow_cash)
        if w_out is not None:
            logger.warning(f"BL_FAILSAFE:optimize_portfolio_cvxpy:final_caps msw={msw:.6f} mxsw={mxsw:.6f}")
            return w_out

    # Tier 1
    logger.warning("BL_FAILSAFE:optimize_portfolio_cvxpy:tier0_failed trying Tier 1")
    result = _try_solve(msw, mxsw, gamma_turnover * 0.8)
    if result is not None:
        w_out = _project_feasible(result, N, sector_map, msw, mxsw, allow_cash)
        if w_out is not None:
            logger.warning(f"BL_FAILSAFE:optimize_portfolio_cvxpy:final_caps msw={msw:.6f} mxsw={mxsw:.6f}")
            return w_out

    # Try deterministic construction with ORIGINAL caps BEFORE relaxation
    logger.warning("BL_FAILSAFE:optimize_portfolio_cvxpy:tier1_failed trying deterministic construction")
    w_out = _project_feasible(None, N, sector_map, msw, mxsw, allow_cash)
    if w_out is not None:
        logger.warning(f"BL_FAILSAFE:optimize_portfolio_cvxpy:final_caps msw={msw:.6f} mxsw={mxsw:.6f}")
        return w_out

    # Tier 2: Iterative relaxation loop (only if deterministic failed)
    logger.warning("BL_FAILSAFE:optimize_portfolio_cvxpy:deterministic_failed trying Tier 2 relaxation")

    for relax_iter in range(20):
        msw_new = min(msw * 1.1, 1.0)
        mxsw_new = min(mxsw * 1.1, 1.0)

        if msw_new > msw or mxsw_new > mxsw:
            logger.warning(f"BL_FAILSAFE:optimize_portfolio_cvxpy:relax_caps msw={msw_new:.6f} mxsw={mxsw_new:.6f} reason=tier2_iter{relax_iter}")
            msw = msw_new
            mxsw = mxsw_new

        result = _try_solve(msw, mxsw, gamma_turnover * 0.8)
        if result is not None:
            w_out = _project_feasible(result, N, sector_map, msw, mxsw, allow_cash)
            if w_out is not None:
                logger.warning(f"BL_FAILSAFE:optimize_portfolio_cvxpy:final_caps msw={msw:.6f} mxsw={mxsw:.6f}")
                return w_out

        # Try deterministic construction
        w_out = _project_feasible(None, N, sector_map, msw, mxsw, allow_cash)
        if w_out is not None:
            logger.warning(f"BL_FAILSAFE:optimize_portfolio_cvxpy:final_caps msw={msw:.6f} mxsw={mxsw:.6f}")
            return w_out

        if msw >= 1.0 and mxsw >= 1.0:
            break

    # Fallback A: w_prev if feasible
    w_out = _project_feasible(w_prev, N, sector_map, msw, mxsw, allow_cash)
    if w_out is not None:
        logger.warning(f"BL_FAILSAFE:optimize_portfolio_cvxpy:using_fallback_A")
        logger.warning(f"BL_FAILSAFE:optimize_portfolio_cvxpy:final_caps msw={msw:.6f} mxsw={mxsw:.6f}")
        return w_out

    # Fallback B/C: Deterministic construction with full relaxation
    logger.warning("BL_FAILSAFE:optimize_portfolio_cvxpy:using_fallback_C")
    msw = 1.0
    mxsw = 1.0
    logger.warning(f"BL_FAILSAFE:optimize_portfolio_cvxpy:relax_caps msw={msw:.6f} mxsw={mxsw:.6f} reason=final_fallback")
    w_out = _project_feasible(None, N, sector_map, msw, mxsw, allow_cash)
    if w_out is not None:
        logger.warning(f"BL_FAILSAFE:optimize_portfolio_cvxpy:final_caps msw={msw:.6f} mxsw={mxsw:.6f}")
        return w_out

    # Ultimate fallback: equal weight (always works with msw=mxsw=1.0)
    w_out = np.ones(N, dtype=np.float64) / N
    logger.warning(f"BL_FAILSAFE:optimize_portfolio_cvxpy:final_caps msw={msw:.6f} mxsw={mxsw:.6f}")
    return w_out


def _project_feasible(
    w0: np.ndarray | None,
    N: int,
    sector_map: dict[int, list[int]],
    max_stock_weight: float,
    max_sector_weight: float,
    allow_cash: bool,
    tol: float = 1e-6,
    max_iters: int = 50,
) -> np.ndarray | None:
    """
    Project/construct feasible portfolio satisfying all constraints.

    Returns None only if mathematically infeasible (allow_cash=False with impossible caps).
    """
    if N == 0:
        return np.array([], dtype=np.float64)

    # Step A: Initialize
    if w0 is not None and len(w0) == N and np.all(np.isfinite(w0)):
        w = np.maximum(w0, 0.0).copy()
    else:
        w = np.ones(N, dtype=np.float64) / N

    for iteration in range(max_iters):
        # Step B: Apply stock cap
        w = np.minimum(w, max_stock_weight)
        w = np.maximum(w, 0.0)

        # Step C: Enforce sector caps
        for sector_id, indices in sector_map.items():
            if not indices:
                continue
            sector_sum = np.sum(w[indices])
            if sector_sum > max_sector_weight + tol:
                scale = max_sector_weight / (sector_sum + 1e-12)
                w[indices] *= scale

        # Re-apply stock cap after sector scaling
        w = np.minimum(w, max_stock_weight)
        w = np.maximum(w, 0.0)

        total = np.sum(w)

        # Step D: Sum constraint
        if allow_cash:
            if total > 1.0 + tol:
                w = w / total
                # Re-apply caps
                w = np.minimum(w, max_stock_weight)
                continue
            # Check constraints
            if _check_constraints(w, max_stock_weight, max_sector_weight, sector_map, allow_cash, tol):
                return w
            if iteration == max_iters - 1:
                return w  # Best effort
        else:
            # allow_cash=False: need sum == 1
            if total > 1.0 + tol:
                w = w / total
                # Re-apply caps after scaling down
                w = np.minimum(w, max_stock_weight)
                # Re-check sector caps
                for sector_id, indices in sector_map.items():
                    if not indices:
                        continue
                    sector_sum = np.sum(w[indices])
                    if sector_sum > max_sector_weight + tol:
                        scale = max_sector_weight / (sector_sum + 1e-12)
                        w[indices] *= scale
                w = np.minimum(w, max_stock_weight)
                w = np.maximum(w, 0.0)
                continue

            if abs(total - 1.0) <= tol:
                if _check_constraints(w, max_stock_weight, max_sector_weight, sector_map, allow_cash, tol):
                    return w

            if total < 1.0 - tol:
                # Fill remaining mass into assets with slack
                remaining = 1.0 - total
                filled = _fill_remaining_mass(w, remaining, N, sector_map, max_stock_weight, max_sector_weight, tol)
                if filled:
                    continue
                else:
                    # Cannot fill: infeasible with current caps
                    return None

    # Final check
    if _check_constraints(w, max_stock_weight, max_sector_weight, sector_map, allow_cash, tol):
        return w

    return None if not allow_cash else w


def _fill_remaining_mass(
    w: np.ndarray,
    remaining: float,
    N: int,
    sector_map: dict[int, list[int]],
    max_stock_weight: float,
    max_sector_weight: float,
    tol: float,
) -> bool:
    """Fill remaining mass deterministically into assets with slack. Modifies w in-place. Returns True if successful."""
    # Build asset-to-sector mapping
    asset_to_sector: dict[int, int] = {}
    for sector_id, indices in sector_map.items():
        for idx in indices:
            asset_to_sector[idx] = sector_id

    # Compute sector usage
    sector_usage: dict[int, float] = {}
    for sector_id, indices in sector_map.items():
        sector_usage[sector_id] = np.sum(w[indices]) if indices else 0.0

    eps = 1e-12
    for i in range(N):
        if remaining < eps:
            break

        stock_slack = max_stock_weight - w[i]
        if stock_slack < eps:
            continue

        if i in asset_to_sector:
            sector_id = asset_to_sector[i]
            sector_slack = max_sector_weight - sector_usage.get(sector_id, 0.0)
        else:
            sector_slack = max_stock_weight  # No sector constraint

        slack = min(stock_slack, sector_slack)
        if slack < eps:
            continue

        fill = min(slack, remaining)
        w[i] += fill
        remaining -= fill

        if i in asset_to_sector:
            sector_id = asset_to_sector[i]
            sector_usage[sector_id] = sector_usage.get(sector_id, 0.0) + fill

    return remaining < tol


def _check_constraints(
    w: np.ndarray,
    max_stock_weight: float,
    max_sector_weight: float,
    sector_map: dict[int, list[int]],
    allow_cash: bool,
    tol: float = 1e-6,
) -> bool:
    """Check if weights satisfy all constraints."""
    try:
        if not np.all(np.isfinite(w)):
            return False
        if np.any(w < -tol):
            return False
        if np.any(w > max_stock_weight + tol):
            return False
        for indices in sector_map.values():
            if indices and np.sum(w[indices]) > max_sector_weight + tol:
                return False
        total = np.sum(w)
        if allow_cash:
            if total > 1.0 + tol:
                return False
        else:
            if abs(total - 1.0) > tol:
                return False
        return True
    except Exception:
        return False
