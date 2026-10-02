"""Independent reference for tests: PyPortfolioOpt's Ledoit-Wolf constant-correlation estimator.

Copied from PyPortfolioOpt 1.6.0, `pypfopt/risk_models.py`,
`CovarianceShrinkage._ledoit_wolf_constant_correlation`, as shipped in the PyPI wheel
pyportfolioopt-1.6.0-py3-none-any.whl (SHA-256
fcdff5dfb4e7d0c477d8b76e9b5aff58defc30370fc3cb096e95dd46ae43d617). It is someone else's reading
of Ledoit and Wolf (2004), used only as a test oracle and never imported by `src/`.

Adapted: the method became a function of `X` (returns, T x N) and `S` (the sample matrix, which
the class computes as `X.cov()`, dividing by T - 1): `self.X.values` became `X`, `self.S` became
`S`, and the docstring was replaced by this one. The computation is unchanged.

MIT License

Copyright (c) 2018 Robert Andrew Martin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np


def pypfopt_ledoit_wolf_constant_correlation(
    X: np.ndarray, S: np.ndarray
) -> tuple[np.ndarray, float]:
    """Shrunk covariance matrix and shrinkage constant, as PyPortfolioOpt 1.6.0 computes them."""
    X = np.nan_to_num(X)
    t, n = np.shape(X)

    # Constant correlation target
    var = np.diag(S).reshape(-1, 1)
    std = np.sqrt(var)
    _var = np.tile(var, (n,))
    _std = np.tile(std, (n,))
    r_bar = (np.sum(S / (_std * _std.T)) - n) / (n * (n - 1))
    F = r_bar * (_std * _std.T)
    F[np.eye(n) == 1] = var.reshape(-1)

    # Estimate pi
    Xm = X - X.mean(axis=0)
    y = Xm**2
    pi_mat = np.dot(y.T, y) / t - 2 * np.dot(Xm.T, Xm) * S / t + S**2
    pi_hat = np.sum(pi_mat)

    # Theta matrix, expanded term by term
    term1 = np.dot((Xm**3).T, Xm) / t
    help_ = np.dot(Xm.T, Xm) / t
    help_diag = np.diag(help_)
    term2 = np.tile(help_diag, (n, 1)).T * S
    term3 = help_ * _var
    term4 = _var * S
    theta_mat = term1 - term2 - term3 + term4
    theta_mat[np.eye(n) == 1] = np.zeros(n)
    rho_hat = sum(np.diag(pi_mat)) + r_bar * np.sum(np.dot((1 / std), std.T) * theta_mat)

    # Estimate gamma
    gamma_hat = np.linalg.norm(S - F, "fro") ** 2

    # Compute shrinkage constant
    kappa_hat = (pi_hat - rho_hat) / gamma_hat
    delta = max(0.0, min(1.0, kappa_hat / t))

    # Compute shrunk covariance matrix
    shrunk_cov = delta * F + (1 - delta) * S
    return shrunk_cov, delta
