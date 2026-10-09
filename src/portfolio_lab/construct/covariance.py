"""Covariance estimators for a window of periodic returns (pure functions).

A window is a DataFrame of simple returns: rows = periods in date order, columns = assets,
every value present and finite. Each estimator returns the per-period covariance matrix (not
annualized), labeled by the assets, and uses nothing but the window it is given.

Conventions:
- The sample and Ledoit-Wolf estimators demean the returns and divide cross-products by T, the
  number of periods, as scikit-learn does (Ledoit and Wolf 2004a divide by T for data with a
  known zero mean). pandas' `cov` divides by T - 1. For the sample matrix that is only a scale,
  which leaves minimum-variance and equal-risk-contribution weights unchanged; inside the
  shrinkage formulas the divisor is part of the estimator: T - 1 would lower the intensity by
  about 2/T in relative terms (0.8% at T = 252, 8% at T = 24).
- EWMA instead uses a zero mean and its own weights (RiskMetrics, below).
- Shrinkage blends the sample matrix S with a structured target F: delta F + (1 - delta) S,
  with the intensity delta in [0, 1]. A target equal to S up to float noise gets delta = 0.

Ledoit and Wolf (2004a, J. Multivariate Analysis), scaled identity target: F = m I with
m = trace(S) / N (the paper's estimate of mu). With ||A||^2 = trace(A A') / N and x_t the
demeaned returns of period t, d^2 = ||S - m I||^2, b^2 = min(d^2, (1 / T^2) sum_t ||x_t x_t' -
S||^2) and delta = b^2 / d^2.

Ledoit and Wolf (2004b, J. Portfolio Management, "Honey, I shrunk the sample covariance
matrix", appendix), constant-correlation target: F_ii = s_ii and F_ij = rbar sqrt(s_ii s_jj),
rbar = the average of the N(N - 1)/2 sample correlations. delta = max(0, min(1, kappa / T)),
kappa = (pi - rho) / gamma, where avg_t averages over the T periods (divides by T):
- pi = sum_ij pi_ij, with pi_ij = avg_t (x_ti x_tj - s_ij)^2;
- rho = sum_i pi_ii + sum_i sum_(j != i) (rbar / 2) (sqrt(s_jj / s_ii) theta_ii,ij
  + sqrt(s_ii / s_jj) theta_jj,ij), with theta_kk,ij = avg_t (x_tk^2 - s_kk)(x_ti x_tj - s_ij);
- gamma = sum_ij (F_ij - s_ij)^2.

EWMA (RiskMetrics): sum_k w_k r_k r_k' with zero mean and w_k proportional to decay^age (age 0
= the latest period), normalized to sum to 1 over the window.

Hand example: returns in %, T = 4, demeaned (the tests add a mean of 1% to a and 2% to c):
    a = 5, -1, -1, -3    b = 1, 1, -1, -1    c = 1, -1, 1, -1
S = [[9, 2, 2], [2, 1, 0], [2, 0, 1]]: sd 3, 1, 1; r_ab = r_ac = 2/3 and r_bc = 0.
- Identity: m = 11/3 and d^2 = 176/9. sum_t ||x_t||^4 = 729 + 9 + 9 + 121 = 868 and
  sum_ij s_ij^2 = 99, so b^2 = (868 - 4 * 99) / (4^2 * 3) = 59/6 and delta = 177/352 = 0.503.
- Constant correlation: rbar = 4/9, so F_ab = F_ac = 4/3 and F_bc = 4/9; gamma = 2 ((2/3)^2 +
  (2/3)^2 + (4/9)^2) = 176/81. pi = 96 + 2 (5 + 5 + 1) = 118. theta_aa,ab = theta_aa,ac = 20
  and every other theta is 0 (b and c have constant squares), so rho = 96 + 4 (rbar / 2) (1/3)
  20 = 2752/27 and kappa = (118 - 2752/27) / (176/81) = 651/88: delta = min(1, 651/352) = 1.
- The same four periods twice (T = 8) leave every average unchanged: delta = 651/704 = 0.925
  for constant correlation and 177/704 = 0.251 for the identity.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_lab._validate import check_decay, window_values

__all__ = [
    "EWMA_DECAY",
    "Shrunk",
    "ewma_cov",
    "ledoit_wolf_constant_correlation",
    "ledoit_wolf_identity",
    "sample_cov",
]

EWMA_DECAY = 0.94  # RiskMetrics' decay for 1-day forecasts from daily returns (1-month: 0.97)

# Relative size below which a difference is float noise: a target this close to the sample
# matrix is the sample matrix, and a variance this small is no variance.
_NOISE = 1e-12


@dataclass(frozen=True, eq=False)  # DataFrame fields: no value equality or hashing
class Shrunk:
    """A shrinkage estimate: cov = intensity * target + (1 - intensity) * sample matrix."""

    cov: pd.DataFrame
    target: pd.DataFrame
    intensity: float


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Demeaned cross-products divided by T (not T - 1)."""
    return _labeled(_cross(_demeaned(_window(returns))), returns)


def ledoit_wolf_identity(returns: pd.DataFrame) -> Shrunk:
    """Ledoit-Wolf (2004a) shrinkage toward trace(S) / N times the identity."""
    x = _demeaned(_window(returns))
    t, n = x.shape
    s = _cross(x)
    target = np.trace(s) / n * np.eye(n)
    # Both squared distances without the 1/N of the paper's norm: it cancels in delta.
    d2 = float(np.sum((s - target) ** 2))
    b2 = float(np.sum(np.sum(x**2, axis=1) ** 2) - t * np.sum(s**2)) / t**2
    is_target = d2 <= _NOISE**2 * np.sum(s**2)
    intensity = 0.0 if is_target else min(max(b2, 0.0), d2) / d2
    return _shrunk(s, target, intensity, returns)


def ledoit_wolf_constant_correlation(returns: pd.DataFrame) -> Shrunk:
    """Ledoit-Wolf (2004b) shrinkage toward the sample variances with one average correlation."""
    values = _window(returns)
    if values.shape[1] < 2:
        raise ValueError("the constant-correlation target needs at least 2 assets")
    x = _demeaned(values)
    s = _cross(x)
    flat = np.diag(s) <= (_NOISE * np.abs(values).max(axis=0)) ** 2
    if flat.any():
        asset = returns.columns[int(np.argmax(flat))]
        raise ValueError(f"{asset} has no variance in the window: its correlations are undefined")
    target, intensity = _lw_cc(x, s)
    return _shrunk(s, target, intensity, returns)


def ewma_cov(returns: pd.DataFrame, *, decay: float = EWMA_DECAY) -> pd.DataFrame:
    """Exponentially weighted covariance with zero mean (RiskMetrics), weights summing to 1."""
    check_decay(decay)
    r = _window(returns)
    w = decay ** np.arange(len(r) - 1, -1, -1, dtype=float)  # age 0 = the latest period
    return _labeled(_symmetric((r * (w / w.sum())[:, None]).T @ r), returns)


def _lw_cc(x: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, float]:
    """Constant-correlation target and intensity from demeaned returns x (T x N) and a sample
    matrix s.

    `s` is an argument so a test can pass a matrix divided by T - 1, as an independent
    implementation does, and compare the two exactly. pi and theta always average with 1/T.
    """
    t, n = x.shape
    sd = np.sqrt(np.diag(s))
    off = ~np.eye(n, dtype=bool)
    rbar = float((s / np.outer(sd, sd))[off].mean())
    target = rbar * np.outer(sd, sd)
    np.fill_diagonal(target, np.diag(s))
    gamma = float(np.sum((target - s) ** 2))
    if gamma <= _NOISE**2 * np.sum(s**2):
        return target, 0.0
    avg = x.T @ x / t  # avg_t x_ti x_tj (equals s when s divides by T)
    pi = (x**2).T @ (x**2) / t - 2 * s * avg + s**2
    s_kk = np.diag(s)[:, None]
    theta = (x**3).T @ x / t - np.diag(avg)[:, None] * s - s_kk * avg + s_kk * s  # theta_ii,ij
    # The paper's two theta terms are mirror images over (i, j) and (j, i): rbar times one sum.
    rho = float(np.trace(pi) + rbar * np.sum((np.outer(1 / sd, sd) * theta)[off]))
    kappa = (float(np.sum(pi)) - rho) / gamma
    return target, min(max(kappa / t, 0.0), 1.0)


def _window(returns: pd.DataFrame) -> np.ndarray:
    return window_values(returns, "returns", min_rows=2)


def _demeaned(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=0)


def _cross(x: np.ndarray) -> np.ndarray:
    """x'x / T, exactly symmetric."""
    return _symmetric(x.T @ x / len(x))


def _symmetric(m: np.ndarray) -> np.ndarray:
    return (m + m.T) / 2


def _shrunk(s: np.ndarray, target: np.ndarray, intensity: float, returns: pd.DataFrame) -> Shrunk:
    cov = intensity * target + (1 - intensity) * s
    return Shrunk(_labeled(cov, returns), _labeled(target, returns), float(intensity))


def _labeled(m: np.ndarray, returns: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(m, index=returns.columns, columns=returns.columns)
