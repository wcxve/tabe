from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded
from scipy.optimize import minimize_scalar

from .l1tf import L1TF
from .utils import (
    create_D,
    create_upper_banded_DTD,
    diag_V_compact,
    huber_weight,
    median_absolute_value,
    relative_difference,
    tukey_weight,
)

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray


class InnerFitResult(NamedTuple):
    """Result of the iteratively re-weighted least squares."""

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

    pcb: NDArray
    """Cholesky factorization of the penalization matrix in banded format."""

    @property
    def diag_H(self) -> NDArray:
        """Diagonal of the hat matrix."""
        return self.w * diag_V_compact(self.pcb)

    @property
    def lnZ(self) -> float:
        """Logarithm of the marginal likelihood."""
        wrss = np.sum(self.w * self.r * self.r)
        yhat_diff = np.diff(self.yhat, self.d)
        m = self.yhat.size - self.d
        # if self.a is None:
        penalty = self.lam * np.sum(yhat_diff * yhat_diff)
        log_det_P = m * np.log(self.lam)
        # else:
        #     penalty = self.lam * np.sum(self.a * yhat_diff * yhat_diff)
        #     log_det_P = m * (np.log(self.lam) + np.sum(np.log(self.a)))
        log_det_WP = 2.0 * np.sum(np.log(self.pcb[self.d]))
        return -0.5 * (wrss + penalty + log_det_WP - log_det_P)


class FitResult(NamedTuple):
    """Result of the iteratively re-weighted least squares."""

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

    # pcb: NDArray
    # """Cholesky factorization of the penalization matrix in banded format."""

    tols: NDArray
    """Relative differences between the iterations."""

    sparse: bool
    """Whether the yhat is obtained from L1 trend filtering."""

    loss: NDArray
    """Loss function values of each iteration of L1 trend filtering."""

    # @property
    # def diag_H(self) -> NDArray:
    #     """Diagonal of the hat matrix."""
    #     return self.w * diag_V_compact(self.pcb)

    # @property
    # def lnZ(self) -> float:
    #     """Logarithm of the marginal likelihood."""
    #     wrss = np.sum(self.w * self.r * self.r)
    #     yhat_diff = np.diff(self.yhat, self.d)
    #     m = self.yhat.size - self.d
    #     # if self.a is None:
    #     penalty = self.lam * np.sum(yhat_diff * yhat_diff)
    #     log_det_P = m * np.log(self.lam)
    #     # else:
    #     #     penalty = self.lam * np.sum(self.a * yhat_diff * yhat_diff)
    #     #     log_det_P = m * (np.log(self.lam) + np.sum(np.log(self.a)))
    #     log_det_WP = 2.0 * np.sum(np.log(self.pcb[self.d]))
    #     return -0.5 * (wrss + penalty + log_det_WP - log_det_P)

    # @property
    # def lnZ_norm(self) -> NDArray:
    #     """Logarithm of the normalized marginal likelihood."""
    #     mask_pos = self.w > 0.0
    #     n_pos = np.sum(mask_pos)
    #     n_neg = self.yhat.size - n_pos
    #     # c^2/6 is Tukey loss's constant
    #     norm = n_pos * np.log(2 * np.pi) + n_neg * c * c / 6
    #     log_W_det = np.log(self.w[mask_pos]).sum()
    #     return self.lnZ - 0.5 * (norm - log_W_det)


class WhittakerSmoother:
    """Whittaker smoother.

    Parameters
    ----------
    y : ndarray
        The data to smooth.
    missing : ndarray, optional
        The mask of missing or bad data.
    d : int, optional
        The order of the difference. Default is 2.
    """

    def __init__(
        self,
        y: NDArray,
        missing: NDArray | None = None,
        d: int = 2,
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
        self.y = self._filter_bad_data(y)
        self.D = create_D(self.n, self.d)
        self.DT = self.D.T
        self.DTD_banded = create_upper_banded_DTD(self.n, self.d)
        self.default_weights = self._filter_bad_data(np.ones(self.n))
        # self.log_det_DTD = np.sum(np.log(eigvals_banded(self.DTD)[self.d:]))

    def _filter_bad_data(self, a: NDArray) -> NDArray:
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

    def _check_w(self, w: NDArray | None) -> NDArray:
        """Check the weights and then set the weights of bad data to 0."""
        if w is None:
            w = self.default_weights
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
        return self._filter_bad_data(w)

    def _check_nit(self, nit: int) -> int:
        """Check the number of iterations."""
        nit = int(nit)
        if nit < 0:
            raise ValueError(f'nit ({nit}) must be non-negative')
        return nit

    def _check_mask(self, mask: NDArray | None) -> NDArray | None:
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
        w: NDArray | None = None,
        a: NDArray | None = None,
        y: NDArray | None = None,
    ) -> InnerFitResult:
        if lam <= 0:
            raise ValueError(f'lam ({lam}) must be positive')

        y = self.y if y is None else y
        d = self.d
        n = self.n

        if a is None:
            pb = lam * self.DTD_banded
        else:
            if a.size != n - d:
                raise ValueError(f'length of a must be {n - d}, got {a.size}')
            DTD = self.DT.multiply(a).dot(self.D)  # D.T @ diags(a) @ D
            pb = np.zeros((d + 1, n))
            for i in range(d + 1):
                k = d - i
                pb[i, k:] = DTD.diagonal(k=k)
            pb *= lam
        if w is None:
            w = self.default_weights
        pb[d] += w
        pcb = cholesky_banded(ab=pb, overwrite_ab=True)
        yhat = cho_solve_banded(
            cb_and_lower=(pcb, False),
            b=w * y,
            overwrite_b=True,
        )
        return InnerFitResult(
            yhat=yhat,
            r=np.where(self.good_mask, y - yhat, 0.0),
            d=d,
            lam=lam,
            # a=a,
            w=w,
            pcb=pcb,
        )

    def fit(
        self,
        lam: float = 1e5,
        weights: NDArray | None = None,
        weight_type: str = 'tukey',
        scale: float | None = None,
        n_neg: int = 5,
        max_iter: int = 50,
        tol: float = 1e-4,
    ) -> FitResult:
        """Perform the Whittaker smoother given a smoothing parameter.

        Parameters
        ----------
        lam : float, optional
            Smoothing parameter. Default is 1e5.
        weights : ndarray, optional
            Weights. Default is None.
        weight_type : str, optional
            Weight function, ``'tukey'`` (default) or ``'huber'``.
        scale : float, optional
            Scale parameter for the weight function.
            Default is 3.0 for ``weight_type='tukey'`` and 1.0 for
            ``weight_type='huber'``.
        n_neg : int, optional
            Weight will be 1.0 for data with `n_neg` consecutive negative
            residuals. Default is 5.
        max_iter : int, optional
            Maximum number of iterations. Default is 50.
        tol : float, optional
            The relative tolerance between the iterations. Default is 1e-4.

        Returns
        -------
        FitResult
            The result of the fit.
        """
        if weight_type == 'tukey':
            weight_fn = tukey_weight
        elif weight_type == 'huber':
            weight_fn = huber_weight
        else:
            raise ValueError(f'{weight_type=} is not implemented')
        if scale is None:
            if weight_type == 'tukey':
                c = 3.0
            elif weight_type == 'huber':
                c = 1.0
        else:
            c = float(scale)
        nit = self._check_nit(max_iter) + 1
        w = self._check_w(weights)
        yhat = self.y
        tols = np.empty(nit)
        for i in range(nit):
            yhat_prev = yhat
            res_i = self._inner_fit(lam=lam, w=w)
            yhat = res_i.yhat
            tols[i] = relative_difference(yhat_prev, yhat)
            if tols[i] <= tol or i == max_iter:
                break
            r = res_i.r
            sigma = median_absolute_value(r[self.good_mask])
            w = weight_fn(r / sigma, c=c, n_neg=n_neg)
            w = self._filter_bad_data(w)

        return FitResult(
            yhat=yhat,
            r=res_i.r,
            d=self.d,
            lam=lam,
            # a=None,
            w=w,
            # pcb=res_i.pcb,
            tols=tols[: i + 1],
            sparse=False,
            loss=np.empty(0),
        )

    def fit_adapt(
        self,
        lam0: float = 1e5,
        weights: NDArray | None = None,
        weight_type: Literal['tukey', 'huber'] = 'tukey',
        scale: float | None = None,
        n_neg: int = 5,
        max_iter0: int = 1,
        max_iter: int = 50,
        tol: float = 1e-4,
        sparse_lam: float = 1e5,
        sparse_max_iter: int = 500,
    ) -> FitResult:
        """Perform the Whittaker smoother with optimal smoothing.

        Parameters
        ----------
        lam0 : float, optional
            Initial smoothing parameter. Default is 1e5.
        weights : ndarray, optional
            Weights. Default is None.
        weight_type : str, optional
            Weight function, ``'tukey'`` (default) or ``'huber'``.
        scale : float, optional
            Scale parameter for the weight function.
            Default is 3.0 for ``weight_type='tukey'`` and 1.0 for
            ``weight_type='huber'``.
        n_neg : int, optional
            Number of consecutive negative residuals to weight 1.0.
            Default is 5.
        max_iter0 : int, optional
            Maximum iterations in initial fit. Default is 1.
        max_iter : int, optional
            Maximum iterations in adapting smoothing parameter. Default is 50.
        tol : float, optional
            The relative tolerance between the iterations. Default is 1e-4.
        sparse_lam : float, optional
            Fit again with L1 trend filtering with the final smoothing
            parameter and weights, if the final smoothing parameter is smaller
            than `sparse_lam`. Default is 1e5.
        sparse_max_iter : int, optional
            Maximum iterations in L1 trend filtering. Default is 500.

        Returns
        -------
        FitResult
            The result of the fit.
        """

        @lru_cache
        def inner_fit(lam_: float) -> InnerFitResult:
            return self._inner_fit(lam_, w)

        def optimize(lam_) -> np.float64:
            """Optimize the smoothing parameter and the adaptive parameter."""
            f = lambda x: -inner_fit(np.power(10.0, x)).lnZ
            lg_lam = np.log10(lam_)
            bounds = (lg_lam - 4, lg_lam + 4)
            res = minimize_scalar(f, bounds=bounds, method='bounded')
            if not res.success:
                raise ValueError(f'Failed to maximize lnZ: {res.message}')
            return np.power(10, res.x)

        def update_weights(
            residuals: NDArray,
        ) -> NDArray:
            """Update the weights."""
            sigma = median_absolute_value(residuals[self.good_mask])
            w = weight_fn(r=residuals / sigma, c=c, n_neg=n_neg)
            return self._filter_bad_data(w)

        if weight_type == 'tukey':
            weight_fn = tukey_weight
        elif weight_type == 'huber':
            weight_fn = huber_weight
        else:
            raise ValueError(f'{weight_type=} is not implemented')
        if scale is None:
            if weight_type == 'tukey':
                c = 3.0
            elif weight_type == 'huber':
                c = 1.0
        else:
            c = float(scale)

        w = self._check_w(weights)
        result = self.fit(
            lam=lam0,
            weights=w,
            weight_type=weight_type,
            scale=c,
            n_neg=n_neg,
            max_iter=max_iter0,
            tol=tol,
        )
        lam = result.lam
        yhat = result.yhat
        r = result.r
        w = update_weights(r)
        max_iter = self._check_nit(max_iter)
        tols = np.empty(max_iter)
        for i in range(max_iter):
            lam = optimize(lam)
            result = inner_fit(lam)
            yhat_prev = yhat
            yhat = result.yhat
            r = result.r
            tols[i] = relative_difference(yhat_prev, yhat)
            if tols[i] <= tol:
                tols = tols[: i + 1]
                break
            w = update_weights(r)

        if sparse_lam > 0.0 and lam < sparse_lam:
            sparse = True
            result_sparse = L1TF(self.y, missing=self.bad_mask, d=self.d).fit(
                lam=lam,
                w=w,
                tol=tol,
                max_iter=sparse_max_iter,
            )
            yhat = result_sparse.yhat
            r = result_sparse.r
            loss = result_sparse.loss
        else:
            sparse = False
            loss = np.empty(0)

        return FitResult(
            yhat=yhat,
            r=r,
            d=self.d,
            lam=lam,
            # a=a,
            w=w,
            # pcb=result.pcb,
            tols=tols,
            sparse=sparse,
            loss=loss,
        )
