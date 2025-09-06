"""Adapted from pybaselines.polynomial.loess"""

import warnings
from math import ceil

import numpy as np
from pybaselines._algorithm_setup import _Algorithm
from pybaselines._compat import _HAS_NUMBA, jit
from pybaselines.polynomial import (
    _determine_fits,
    _loess_solver,
    _median_absolute_value,
    _tukey_square,
)
from pybaselines.utils import (
    ParameterWarning,
    _convert_coef,
    relative_difference,
)


class LOESS(_Algorithm):
    @_Algorithm._register(sort_keys=('weights', 'coef'), require_unique_x=True)
    def fit(
        self,
        data,
        fraction=0.2,
        total_points=None,
        poly_order=1,
        scale=3.0,
        tol=1e-3,
        max_iter=50,
        symmetric_weights=True,
        use_threshold=True,
        num_std=1,
        use_original=True,
        weights=None,
        return_coef=False,
        conserve_memory=True,
        mask=None,
        mask_total_points=None,
        mask_poly_order=None,
    ):
        """
        Locally estimated scatterplot smoothing (LOESS).

        Performs polynomial regression at each data point using the nearest
        points.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        fraction : float, optional
            The fraction of N data points to include for the fitting on each
            point.
            Default is 0.2. Not used if `total_points` is not None.
        total_points : int, optional
            The total number of points to include for the fitting on each
            point. Default is None, which will use `fraction` * N to determine
            the number of points.
        scale : float, optional
            A scale factor applied to the weighted residuals to control the
            robustness of the fit. Default is 3.0, as used in [1]_. Note that
            the original loess procedure for smoothing in [2]_ used a `scale`
            of ~4.05.
        poly_order : int, optional
            The polynomial order for fitting the baseline. Default is 1.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        max_iter : int, optional
            The maximum number of iterations. Default is 50.
        symmetric_weights : bool, optional
            If True (default), will apply weighting the same for both positive
            and negative residuals, which is regular LOESS. If False, will
            apply weighting asymmetrically, with residuals < 0 having a weight
            of 1, according to [1]_.
            If `use_threshold` is True, this parameter is ignored.
        use_threshold : bool, optional
            If True (default), will apply a threshold on the data being fit
            each iteration, based on the maximum values of the data and the fit
            baseline, as proposed by [3]_, similar to the modpoly and imodpoly
            techniques. If False, will compute weights each iteration to
            perform the robust fitting, which is regular LOESS.
        num_std : float, optional
            The number of standard deviations to include when thresholding.
            Default is 1, which is the value used for the imodpoly technique.
            Only used if `use_threshold` is True.
        use_original : bool, optional
            If True (default), will compare the baseline with the original
            y-values given by `data` [5]_. If False, will compare the baseline
            of each iteration with the y-values of that iteration [4]_ when
            choosing minimum values for thresholding. Only used if
            `use_threshold` is True.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then will be an array with
            size equal to N and all values set to 1.
        return_coef : bool, optional
            If True, will convert the polynomial coefficients for the fit
            baseline to a form that fits the input x_data and return them in
            the params dictionary. Default is False, since the conversion takes
            time.
        conserve_memory : bool, optional
            If False, will cache the distance-weighted kernels for each value
            in `x_data` on the first iteration and reuse them on subsequent
            iterations to save time. The shape of the array of kernels is
            (len(`x_data`), `total_points`). If True (default), will
            recalculate the kernels each iteration, which uses very little
            memory, but is slower. Can usually set to False unless `x_data`
            and`total_points` are quite large and the function causes memory
            issues when caching the kernels. If numba is installed, there is
            no significant time difference since the calculations are sped up.
        mask : array-like, shape (N,), optional
            The boolean array of the same size as `data` that indicates which
            points use different fitting parameters. Default is None, which
            means all points use the same fitting parameters.
        mask_total_points : int, optional
            The total number of points to include for the fitting on each
            masked point. Default is None, which will use `fraction` * N to
            determine the number of points.
        mask_poly_order : int, optional
            The polynomial order for fitting the baseline on masked points.
            Default is as `poly_order`.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data. Does NOT contain
                the individual distance-weighted kernels for each x-value.
            * 'residuals': numpy.ndarray, shape (N,)
                The residuals of the fit.
            * 'diag_H': float
                The diagonal of the hat matrix.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of
                iterations completed. If the last value in the array is greater
                than the input `tol` value, then the function did not converge.
            * 'aicc': float
                The Akaike information criterion corrected for small sample
                size.
            * 'gcv': float
                The generalized cross-validation score.
            * 'coef': numpy.ndarray, shape (N, poly_order + 1)
                Only if `return_coef` is True. The array of polynomial
                parameters for the baseline, in increasing order. Can be used
                to create a polynomial using
                :class:`numpy.polynomial.polynomial.Polynomial`. If `delta` is
                > 0, the coefficients for any skipped x-value will all be 0.

        Raises
        ------
        ValueError
            Raised if the number of points per window for the fitting is less
            than `poly_order` + 1 or greater than the total number of points,
            or if the values in `self.x` are not strictly increasing.

        Notes
        -----
        The iterative, robust, aspect of the fitting can be achieved either
        through reweighting based on the residuals (the typical usage), or
        thresholding the fit data based on the residuals, as proposed by [3]_,
        similar to the modpoly and imodpoly techniques.

        In baseline literature, this procedure is sometimes called "rbe",
        meaning "robust baseline estimate".

        References
        ----------
        .. [1] Ruckstuhl, A.F., et al. Baseline subtraction using robust local
                regression estimation. J. Quantitative Spectroscopy and
                Radiative Transfer, 2001, 68, 179-193.
        .. [2] Cleveland, W. Robust locally weighted regression and smoothing
                scatterplots. Journal of the American Statistical Association,
                1979, 74(368), 829-836.
        .. [3] Komsta, Ł. Comparison of Several Methods of Chromatographic
                Baseline Removal with a New Approach Based on Quantile
                Regression. Chromatographia, 2011, 73, 721-731.
        .. [4] Gan, F., et al. Baseline correction by improved iterative
                polynomial fitting with automatic threshold. Chemometrics and
                Intelligent Laboratory Systems, 2006, 82, 59-65.
        .. [5] Lieber, C., et al. Automated method for subtraction of
                fluorescence from biological raman spectra. Applied
                Spectroscopy, 2003, 57(11), 1363-1367.
        .. [6] https://github.com/statsmodels/statsmodels.
        .. [7] https://www.netlib.org/go (lowess.f is the file).

        """
        if total_points is None:
            total_points = ceil(fraction * self._size)
        if total_points < poly_order + 1:
            raise ValueError(
                'total points must be greater than polynomial order + 1'
            )
        elif total_points > self._size:
            raise ValueError(
                'points per window is higher than total number of points; '
                'lower either "fraction" or "total_points"'
            )
        elif poly_order > 2:
            warnings.warn(
                'polynomial orders greater than 2 can have numerical issues;'
                ' consider using a polynomial order of 1 or 2 instead',
                ParameterWarning,
                stacklevel=2,
            )

        if mask_total_points is None:
            mask_total_points = total_points
        if mask_poly_order is None:
            mask_poly_order = poly_order
        if mask_total_points < mask_poly_order + 1:
            raise ValueError(
                'mask_total_points must be greater than mask_poly_order + 1'
            )
        elif mask_total_points > self._size:
            raise ValueError(
                'mask_total_points is greater than total number of points; '
                'lower either "mask_fraction" or "mask_total_points"'
            )
        elif mask_poly_order > 2:
            warnings.warn(
                'polynomial orders greater than 2 can have numerical issues;'
                ' consider using a polynomial order of 1 or 2 instead',
                ParameterWarning,
                stacklevel=2,
            )

        y, weight_array = self._setup_polynomial(
            data, weights, poly_order, calc_vander=True
        )
        y0 = y

        # x is the scaled version of self.x to fit within the [-1, 1] domain
        x = np.polynomial.polyutils.mapdomain(
            self.x, self.x_domain, np.array([-1.0, 1.0])
        )
        # find the indices for fitting beforehand so that the fitting can be
        # done in parallel; cast delta as float so numba does not have to
        # compile for both int and float
        windows, _, _ = _determine_fits(self.x, self._size, total_points, 0.0)

        # np.polynomial.polynomial.polyvander returns a Fortran-ordered array,
        # which is not continguous when indexed (ie. vandermonde[i]) and
        # issues a warning when using numba, so convert Vandermonde matrix to
        # C-ordering; without Numba, there is no major slowdown using the
        # non-contiguous array
        if _HAS_NUMBA:
            vandermonde = np.ascontiguousarray(self._polynomial.vandermonde)
        else:
            vandermonde = self._polynomial.vandermonde

        if mask is not None:
            if mask.size != self._size:
                raise ValueError('mask must be the same size as data')
            windows_mask, _, _ = _determine_fits(
                self.x, self._size, mask_total_points, 0.0
            )
            windows[mask] = windows_mask[mask]
            self._setup_polynomial(data, weights, poly_order, calc_vander=True)
            vander_mask = self._polynomial.vandermonde
            if _HAS_NUMBA:
                vander_mask = np.ascontiguousarray(vander_mask)
        else:
            mask = np.zeros(self._size, dtype=bool)
            vander_mask = vandermonde

        baseline = y
        coefs = np.zeros((self._size, max(poly_order, mask_poly_order) + 1))
        tol_history = np.empty(max_iter + 1)
        sqrt_w = np.sqrt(weight_array)
        # do max_iter + 1 since a max_iter of 0 would return y as baseline
        # otherwise
        for i in range(max_iter + 1):
            baseline_old = baseline
            if conserve_memory:
                baseline = _loess_low_memory(
                    x,
                    y,
                    sqrt_w,
                    mask,
                    coefs,
                    vandermonde,
                    vander_mask,
                    self._size,
                    windows,
                )
            elif i == 0:
                kernels, kernels_mask, baseline = _loess_first_loop(
                    x,
                    y,
                    sqrt_w,
                    mask,
                    coefs,
                    vandermonde,
                    vander_mask,
                    total_points,
                    mask_total_points,
                    self._size,
                    windows,
                )
            else:
                baseline = _loess_nonfirst_loops(
                    y,
                    sqrt_w,
                    mask,
                    coefs,
                    vandermonde,
                    vander_mask,
                    kernels,
                    kernels_mask,
                    windows,
                    self._size,
                )
            # if i == 0:
            #     diag_H0 = _loess_diag_H(
            #         x,
            #         sqrt_w,
            #         vandermonde,
            #         self._size,
            #         windows,
            #         fits,
            #     )

            calc_difference = relative_difference(baseline_old, baseline)
            tol_history[i] = calc_difference
            residual = y0 - baseline
            if calc_difference < tol:
                break

            if use_threshold:
                y_ = y0 if use_original else y
                residual_ = residual if use_original else y_ - baseline
                y = np.minimum(
                    y_,
                    baseline + num_std * np.std(residual_),
                )
            else:
                # TODO median_absolute_value can be 0 if more than half of
                # residuals are 0 (perfect fit); can that ever really happen?
                # if so, should prevent dividing by 0
                sqrt_w = _tukey_square(
                    residual / _median_absolute_value(residual),
                    scale,
                    symmetric_weights,
                )

        # if use_threshold:
        #     sqrt_w = _tukey_square(
        #         residual / _median_absolute_value(residual),
        #         scale,
        #         symmetric_weights,
        #     )
        # diag_H = _loess_diag_H(
        #     x,
        #     sqrt_w,
        #     vandermonde,
        #     self._size,
        #     windows,
        #     fits,
        # )

        # n = len(residual)
        # tr_H = np.sum(diag_H)
        weights = sqrt_w * sqrt_w
        # sigma2 = np.sum(weights * residual * residual) / n
        params = {
            'weights': weights,
            'residuals': residual,
            # 'diag_H0': diag_H0,
            # 'diag_H': diag_H,
            'tol_history': tol_history[: i + 1],
            # 'aicc': np.log(sigma2) + 2 * (tr_H + 1) / (n - tr_H - 2),
            # 'gcv': sigma2 / np.square(1 - tr_H / n),
        }
        if return_coef:
            # TODO maybe leave out the coefficients from the rest of the
            # calculations since they are otherwise unused, and just fit x vs
            # baseline here; would save a little memory; is providing
            # coefficients for loess even useful?
            params['coef'] = np.array(
                [_convert_coef(coef, self.x_domain) for coef in coefs]
            )

        return baseline, params


def _tukey_weight(residual, scale=3, symmetric=False):
    """Tukey's weight function."""
    r = residual
    u = r / scale
    sqrt_w = np.maximum(1 - u * u, 0)
    if not symmetric:
        mask = r < 0
        sqrt_w[mask] = 1.0
    return sqrt_w * sqrt_w


def _tukey_psi(residual, scale=3, symmetric=False):
    """First order derivative of Tukey's loss function."""
    r = residual
    u = r / scale
    sqrt_w = np.maximum(1 - u * u, 0)
    if not symmetric:
        mask = r < 0
        sqrt_w[mask] = 1.0
    return r * sqrt_w * sqrt_w


def _tukey_Psi(residual, scale=3, symmetric=False):
    """Second order derivative of Tukey's loss function."""
    u = residual / scale
    u2 = u * u
    out = (1 - u2) * (1 - 5 * u2)
    out = np.where(u2 < 1, out, 0.0)
    if not symmetric:
        mask = u < 0.0
        out[mask] = 0.0
    return out


# adapted from (https://gist.github.com/agramfort/850437); see license above
@jit(nopython=True, cache=True)
def _loess_low_memory(
    x, y, weights, mask, coefs, vander, vander_mask, num_x, windows
):
    """
    A version of loess that uses near constant memory.

    The distance-weighted kernel for each x-value is computed each loop, rather
    than cached, so memory usage is low but the calculation is slightly slower.

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
        The x-values of the measured data, with N data points.
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, with N points.
    weights : numpy.ndarray, shape (N,)
        The array of weights.
    mask : numpy.ndarray, shape (N,)
        The boolean array of the same size as `data` that indicates which
        points use different fitting parameters.
    coefs : numpy.ndarray, shape (N, ``poly_order + 1``)
        The array of polynomial coefficients (with polynomial order
        poly_order), for each value in `x`.
    vander : numpy.ndarray, shape (N, ``poly_order + 1``)
        The Vandermonde matrix for the `x` array.
    vander_mask : numpy.ndarray, shape (N, ``poly_order + 1``)
        The Vandermonde matrix for the `x` array for the masked points.
    num_x : int
        The number of data points in `x`, also known as N.
    windows : numpy.ndarray, shape (N, 2)
        An array of left and right indices that define the fitting window for
        each fit x-value.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.

    Notes
    -----
    The coefficient array, `coefs`, is modified inplace.

    """
    baseline = np.empty(num_x)
    y_fit = y * weights
    vander_fit = vander.T * weights
    vander_fit_mask = vander_mask.T * weights
    p = vander_fit.shape[0]
    p_mask = vander_fit_mask.shape[0]
    for i in range(num_x):
        window = windows[i]
        left = window[0]
        right = window[1]

        difference = np.abs(x[left:right] - x[i])
        difference = difference / max(difference[0], difference[-1])
        difference = difference * difference * difference
        difference = 1 - difference
        kernel = np.sqrt(difference * difference * difference)
        if mask[i]:
            vander_fit_i = vander_fit_mask
            p_i = p_mask
        else:
            vander_fit_i = vander_fit
            p_i = p
        AT = kernel * vander_fit_i[:, left:right]
        b = kernel * y_fit[left:right]
        try:
            coef = _loess_solver(AT, b)
        except Exception:
            coef = np.linalg.lstsq(AT.T, b)[0]
        baseline[i] = vander[i].dot(coef)
        coefs[i][:p_i] = coef

    return baseline


# adapted from (https://gist.github.com/agramfort/850437); see license above
@jit(nopython=True, cache=True)
def _loess_first_loop(
    x,
    y,
    weights,
    mask,
    coefs,
    vander,
    vander_mask,
    total_points,
    total_points_mask,
    num_x,
    windows,
):
    """
    The initial fit for loess that also caches the window values for each x.

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
        The x-values of the measured data, with N data points.
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, with N points.
    weights : numpy.ndarray, shape (N,)
        The array of weights.
    mask : numpy.ndarray, shape (N,)
        The boolean array of the same size as `data` that indicates which
        points use different fitting parameters.
    coefs : numpy.ndarray, shape (N, ``poly_order + 1``)
        The array of polynomial coefficients (with polynomial order
        poly_order), for each value in `x`.
    vander : numpy.ndarray, shape (N, ``poly_order + 1``)
        The Vandermonde matrix for the `x` array.
    vander_mask : numpy.ndarray, shape (N, ``poly_order + 1``)
        The Vandermonde matrix for the `x` array for the masked points.
    total_points : int
        The number of points to include when fitting each x-value.
    total_points_mask : int
        The number of points to include when fitting each masked x-value.
    num_x : int
        The number of data points in `x`, also known as N.
    windows : numpy.ndarray, shape (N, 2)
        An array of left and right indices that define the fitting window for
        each fit x-value.

    Returns
    -------
    kernels : numpy.ndarray, shape (N, total_points)
        The array containing the distance-weighted kernel for each x-value.
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.

    Notes
    -----
    The coefficient array, `coefs`, is modified inplace.

    """
    kernels = np.empty((num_x, total_points))
    kernels_mask = np.empty((num_x, total_points_mask))
    baseline = np.empty(num_x)
    y_fit = y * weights
    vander_fit = vander.T * weights
    vander_fit_mask = vander_mask.T * weights
    p = vander_fit.shape[0]
    p_mask = vander_fit_mask.shape[0]
    for i in range(num_x):
        window = windows[i]
        left = window[0]
        right = window[1]

        difference = np.abs(x[left:right] - x[i])
        difference = difference / max(difference[0], difference[-1])
        difference = difference * difference * difference
        difference = 1 - difference
        kernel = np.sqrt(difference * difference * difference)
        if mask[i]:
            kernels_mask[i] = kernel
            vander_fit_i = vander_fit_mask
            p_i = p_mask
        else:
            kernels[i] = kernel
            vander_fit_i = vander_fit
            p_i = p
        AT = kernel * vander_fit_i[:, left:right]
        b = kernel * y_fit[left:right]
        try:
            coef = _loess_solver(AT, b)
        except Exception:
            coef = np.linalg.lstsq(AT.T, b)[0]
        baseline[i] = vander[i].dot(coef)
        coefs[i][:p_i] = coef

    return kernels, kernels_mask, baseline


@jit(nopython=True, cache=True)
def _loess_nonfirst_loops(
    y,
    weights,
    mask,
    coefs,
    vander,
    vander_mask,
    kernels,
    kernels_mask,
    windows,
    num_x,
):
    """
    The loess fit to use after the first loop that uses the cached window.

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, with N points.
    weights : numpy.ndarray, shape (N,)
        The array of weights.
    mask : numpy.ndarray, shape (N,)
        The boolean array of the same size as `data` that indicates which
        points use different fitting parameters.
    coefs : numpy.ndarray, shape (N, ``poly_order + 1``)
        The array of polynomial coefficients (with polynomial order
        poly_order), for each value in `x`.
    vander : numpy.ndarray, shape (N, ``poly_order + 1``)
        The Vandermonde matrix for the `x` array.
    vander_mask : numpy.ndarray, shape (N, ``poly_order + 1``)
        The Vandermonde matrix for the `x` array for the masked points.
    kernels : numpy.ndarray, shape (N, total_points)
        The array containing the distance-weighted kernel for each x-value.
        Each kernel has a length of total_points.
    windows : numpy.ndarray, shape (N, 2)
        An array of left and right indices that define the fitting window for
        each fit x-value.
    num_x : int
        The total number of values, N.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.

    Notes
    -----
    The coefficient array, `coefs`, is modified inplace.

    """
    baseline = np.empty(num_x)
    y_fit = y * weights
    vander_fit = vander.T * weights
    vander_fit_mask = vander_mask.T * weights
    p = vander_fit.shape[0]
    p_mask = vander_fit_mask.shape[0]
    for i in range(num_x):
        window = windows[i]
        left = window[0]
        right = window[1]
        if mask[i]:
            kernel = kernels_mask[i]
            vander_fit_i = vander_fit_mask
            p_i = p_mask
        else:
            kernel = kernels[i]
            vander_fit_i = vander_fit
            p_i = p
        AT = kernel * vander_fit_i[:, left:right]
        b = kernel * y_fit[left:right]
        try:
            coef = _loess_solver(AT, b)
        except Exception:
            coef = np.linalg.lstsq(AT.T, b)[0]
        baseline[i] = vander[i].dot(coef)
        coefs[i][:p_i] = coef

    return baseline


@jit(nopython=True, cache=True)
def _loess_diag_H(x, weights, vander, num_x, windows, fits):
    vander_fit = vander.T * weights
    diag_H = np.empty(num_x)
    for idx in range(fits.shape[0]):
        i = fits[idx]
        window = windows[idx]
        left = window[0]
        right = window[1]
        difference = np.abs(x[left:right] - x[i])
        difference = difference / max(difference[0], difference[-1])
        difference = difference * difference * difference
        difference = 1 - difference
        kernel = np.sqrt(difference * difference * difference)
        AT = kernel * vander_fit[:, left:right]
        Q, _ = np.linalg.qr(AT.T)
        j = i - left
        diag_H[i] = np.sum(Q[j] * Q[j])
    return diag_H
