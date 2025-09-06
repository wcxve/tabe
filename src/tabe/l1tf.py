from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from prox_tv import tv1_1d
from scipy.linalg import cho_solve_banded, cholesky_banded

from .utils import (
    create_D,
    create_upper_banded_DTD,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class L1TFResult(NamedTuple):
    """Result of the L1 trend filtering."""

    yhat: NDArray
    """Trend of the data."""

    r: NDArray
    """Residuals."""

    d: int
    """Order of the difference."""

    lam: float
    """Smoothing parameter."""

    w: NDArray
    """Weights."""

    df: int
    """Degree of freedom."""

    wr2: float
    """Weighted residual sum of squares."""

    loss: NDArray
    """Loss function values of each iteration."""

    @property
    def sure(self) -> float:
        return self.wr2 + 2 * self.df

    @property
    def bic(self) -> float:
        return self.wr2 + np.log(np.sum(self.w > 0)) * self.df


class L1TF:
    """L1 Trend Filtering.

    Parameters
    ----------
    y : ndarray
        The data to smooth.
    missing : ndarray, optional
        The mask of missing or bad data.
    d : int, optional
        The order of the difference. Default is 3.
    """

    def __init__(
        self,
        y: NDArray[np.float64],
        missing: NDArray[np.bool_] | None = None,
        d: int = 3,
    ):
        y = np.array(y, dtype=np.float64)
        if missing is None:
            missing = np.full(y.shape, False)
        else:
            missing = np.array(missing, dtype=np.bool_)
            if missing.shape != y.shape:
                raise ValueError(
                    f'missing must have shape {y.shape}, got {missing.shape}'
                )

        bad_mask = np.logical_not(np.isfinite(y))
        self.bad_mask = np.logical_or(bad_mask, missing)
        self.good_mask = np.logical_not(self.bad_mask)
        if np.sum(self.bad_mask) == len(y):
            raise ValueError('no valid data')
        self.n = len(y)
        self.d = int(d)
        if self.n <= self.d:
            raise ValueError(
                f'y length ({self.n}) must be greater than d ({self.d})'
            )
        self.y = self._filter_bad_data(y)
        self.D = create_D(self.n, self.d)
        self.DT = self.D.T
        self.DTD_banded = create_upper_banded_DTD(self.n, self.d)
        self._default_weights = np.ones(self.n)

    def _filter_bad_data(self, a: NDArray[np.float64]) -> NDArray[np.float64]:
        """Filter bad data.

        Parameters
        ----------
        a : ndarray
            The array to filter.

        Returns
        -------
        a : ndarray
            The filtered array.
        """
        if a.shape != (self.n,):
            raise ValueError(f'shape of a {a.shape} must be ({self.n},)')
        a[self.bad_mask] = 0.0
        return a

    def _check_w(self, w: NDArray[np.float64] | None) -> NDArray[np.float64]:
        """Check the weights and then set the weights of bad data to 0."""
        if w is None:
            w = np.ones(self.n)
        else:
            w = np.asarray(w, dtype=np.float64)
            if w.shape != self.y.shape:
                raise ValueError(
                    f'weights must have shape {self.y.shape}, got {w.shape}'
                )
            if np.any(w < 0.0):
                raise ValueError('weights must be non-negative')
            if not np.any(w > 0.0):
                raise ValueError('weights must be not all zero')
        w = self._filter_bad_data(w)
        if not np.any(w > 0.0):
            raise ValueError('weights are all zero after masking bad data')
        return w

    def _check_nit(self, nit: int) -> int:
        """Check the number of iterations."""
        nit = int(nit)
        if nit < 0:
            raise ValueError(f'nit ({nit}) must be non-negative')
        return nit

    def _check_mask(self, mask: NDArray | None) -> NDArray[np.bool_] | None:
        """Check the mask."""
        if mask is None:
            return None
        mask = np.asarray(mask, dtype=np.bool_)
        if mask.shape != self.y.shape:
            raise ValueError(
                f'mask must have shape {self.y.shape}, got {mask.shape}'
            )
        return None if not np.any(mask) else mask

    def _inner_fit(
        self,
        lam: float,
        w: NDArray[np.float64] | None = None,
        rho: float | None = None,
        tol: float = 1e-6,
        max_iter: int = 500,
        y: NDArray[np.float64] | None = None,
    ) -> L1TFResult:
        if lam <= 0:
            raise ValueError(f'lam ({lam}) must be positive')

        if rho is None:
            rho = lam
        elif rho <= 0:
            raise ValueError(f'rho ({rho}) must be positive')

        tf_dp_lam = lam / rho

        # Validate and apply masking to weights (zeros out bad/missing samples)
        w = self._check_w(w)
        # Validate/prepare y
        if y is None:
            y = self.y
        else:
            y = np.asarray(y, dtype=np.float64)
            if y.shape != self.y.shape:
                raise ValueError(
                    f'y must have shape {self.y.shape}, got {y.shape}'
                )
            y = self._filter_bad_data(y)
        wy = w * y
        lhs = rho * self.DTD_banded
        lhs[-1] += w
        lhs = cholesky_banded(ab=lhs, overwrite_ab=True)

        a = self.D.dot(y)
        u = np.zeros_like(a)

        yhat = np.zeros_like(y)
        f_best = np.inf
        atol = f_best * tol
        loss = np.empty(max_iter)
        for i in range(max_iter):
            # yhat_prev = yhat
            rhs = wy + rho * self.DT.dot(a + u)
            yhat = cho_solve_banded(
                cb_and_lower=(lhs, False),
                b=rhs,
                overwrite_b=True,
            )
            D_yhat = np.diff(yhat, self.d)
            a = tv1_1d(D_yhat - u, tf_dp_lam)
            # from skimage.restoration import denoise_tv_chambolle
            # a = denoise_tv_chambolle(D_yhat - u, tf_dp_lam)
            u += a - D_yhat
            # tols[i] = relative_difference(yhat_prev, yhat)
            # if tols[i] < tol:
            #     tols = tols[: i + 1]
            #     break
            r = y - yhat
            wr2 = np.sum(w * r * r)
            f = wr2 + lam * np.linalg.norm(D_yhat, 1)
            loss[i] = f
            if f < f_best:
                delta_f = f_best - f
                atol = f_best * tol
                f_best = f
                a_best = a
                yhat_best = yhat
                r_best = r
                if i and delta_f < atol:
                    loss = loss[: i + 1]
                    break
            if i > 10:
                variation = np.abs(np.diff(loss[i - 9 : i + 1])).sum()
                if variation < 10 * atol:
                    loss = loss[: i + 1]
                    break
        # yhat_best = yhat
        # r_best = y - yhat
        # a_best = a
        df = np.sum(a_best[:-1] != a_best[1:])

        return L1TFResult(
            yhat_best,
            r_best,
            self.d,
            lam,
            w,
            df,
            np.sum(w * r_best * r_best),
            loss,
        )

    def fit(
        self,
        lam: float,
        w: NDArray[np.float64] | None = None,
        rho: float | None = None,
        tol: float = 1e-6,
        max_iter: int = 500,
    ) -> L1TFResult:
        return self._inner_fit(lam, w, rho, tol, max_iter)
