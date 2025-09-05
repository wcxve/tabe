from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from scipy.sparse import diags

if TYPE_CHECKING:
    from numpy.typing import NDArray


def anscombe_transform(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Anscombe transform."""
    return 2 * np.sqrt(x + 0.375)


def inverse_anscombe_transform(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of the Anscombe transform.

    References
    ----------
    .. [1] M. Makitalo and A. Foi, "A Closed-Form Approximation of the Exact
       Unbiased Inverse of the Anscombe Variance-Stabilizing Transformation",
       doi: 10.1109/TIP.2011.2121085.
    .. [2] M. Makitalo and A. Foi, "Optimal Inversion of the Generalized
       Anscombe Transformation for Poisson-Gaussian Noise",
       doi: 10.1109/TIP.2012.2202675.
    """
    y2 = y * y
    y3 = y2 * y
    sqrt3_2 = np.sqrt(1.5)
    return (
        0.25 * y2
        - 0.125
        + 0.25 * sqrt3_2 / y
        - 1.375 / y2
        + 0.625 * sqrt3_2 / y3
    )


def create_D(n: int, order: int = 2):
    """Create a difference matrix D of order `order`.

    Parameters
    ----------
    n : int
        Size of the matrix.
    order : int, optional
        Order of the difference. Default is 2.

    Returns
    -------
    D : scipy.sparse.csr_matrix
        Difference matrix.
    """
    m = n - order
    if order == 1:
        coeffs = [-1.0, 1.0]
    elif order == 2:
        coeffs = [1.0, -2.0, 1.0]
    elif order == 3:
        coeffs = [-1.0, 3.0, -3.0, 1.0]
    else:
        raise NotImplementedError(f'{order=} is not implemented')
    diagonals = [np.full(m, coeff) for coeff in coeffs]
    offsets = list(range(order + 1))
    return diags(diagonals, offsets, shape=(m, n), format='csr')


def create_upper_banded_DTD(n: int, d: int = 2) -> NDArray[np.float64]:
    """Construct the upper banded matrix of :math:`D^T D`."""
    if d != 2:
        D = create_D(n, d)
        DTD = D.T @ D
        b = np.zeros((d + 1, n))
        for i in range(d + 1):
            k = d - i
            b[i, k:] = DTD.diagonal(k=k)
        return b

    DTD_main = np.ones(n)
    DTD_main[1] = 5
    DTD_main[2:-2] = 6
    DTD_main[-2] = 5
    DTD_main[-1] = 1

    DTD_upper1 = np.zeros(n - 1)
    DTD_upper1[0] = -2
    DTD_upper1[1:-1] = -4
    DTD_upper1[-1] = -2

    DTD_upper2 = np.ones(n - 2)

    main_diag = DTD_main
    off_diag1 = DTD_upper1
    off_diag2 = DTD_upper2

    # banded matrix format
    b = np.zeros((3, n))
    b[0, 2:] = off_diag2
    b[1, 1:] = off_diag1
    b[2, :] = main_diag

    return b


@njit('float64[::1](float64[:,::1])', fastmath=True)
def diag_V_compact(C: np.ndarray) -> np.ndarray:
    """
    Compute diagonal of V = inv(C)' * inv(C) for a banded triangular matrix.

    Parameters
    ----------
    C : np.ndarray
        Upper triangular banded matrix.

    Returns
    -------
    np.ndarray
        Diagonal elements of V
    """
    p = C.shape[1]
    q = C.shape[0] - 1

    result = np.zeros(p)
    K_col = np.zeros(p)
    K_diag = np.zeros(p)

    # Compute diagonal of K = inv(C)
    for i in range(p):
        Kii = 1.0 / C[q, i]
        K_diag[i] = Kii
        result[i] = Kii * Kii

    # Compute off-diagonal contributions
    for j in range(p - 1, -1, -1):
        K_col[j] = K_diag[j]

        for i in range(j - 1, -1, -1):
            delta = i + q
            sum_val = 0.0

            k_max = min(j, delta)
            for k in range(i + 1, k_max + 1):
                sum_val += C[delta - k, k] * K_col[k]

            Kij = -K_diag[i] * sum_val
            K_col[i] = Kij
            result[i] += Kij * Kij

    return result


def relative_difference(
    old: NDArray[np.float64],
    new: NDArray[np.float64],
) -> np.float64:
    """Calculates the relative difference, ``(norm(new-old) / norm(old))``.

    Parameters
    ----------
    old : ndarray
        The array or single value from the previous iteration.
    new : ndarray
        The array or single value from the current iteration.

    Returns
    -------
    float64
        The relative difference between the old and new values.
    """
    numerator = np.linalg.norm(new - old)
    denominator = np.maximum(np.linalg.norm(old), np.finfo(float).eps)
    return numerator / denominator


def mark_consecutive_negatives(arr: NDArray, n: int) -> NDArray[np.bool_]:
    """Mark the `n` consecutive negative values in the array.

    Parameters
    ----------
    arr : ndarray
        The array to mark.
    n : int
        The number of consecutive negative values to mark.

    Returns
    -------
    result : ndarray
        The array with the n consecutive negative values marked.
    """
    arr = np.asarray(arr)
    neg_mask = arr < 0
    window = np.ones(n, dtype=int)
    conv = np.convolve(neg_mask, window, mode='valid')
    idx = conv >= n
    result = np.zeros_like(arr, dtype=bool)
    for i in np.where(idx)[0]:
        result[i : i + n] = True
    return result


def huber_weight(
    r: NDArray[np.float64],
    c: float = 1.345,
    n_neg: int = 0,
) -> NDArray[np.float64]:
    """Huber weight function.

    Parameters
    ----------
    r : ndarray
        Residuals.
    c : float, optional
        Scale parameter. Default is 1.345.
    n_neg : int, optional
        Weight will be 1.0 for `n_neg` consecutive negative residuals.

    Returns
    -------
    w : ndarray
        Weight values.
    """
    r_abs = np.abs(r)
    w = c / np.where(r_abs <= c, c, r_abs)
    n_neg = int(n_neg)
    if n_neg > 0:
        if n_neg == 1:
            mask = r < 0.0
        else:
            mask = mark_consecutive_negatives(r, n_neg)
        w[mask] = 1.0
    return w


def tukey_weight(
    r: NDArray[np.float64],
    c: float = 4.685,
    n_neg: int = 0,
) -> NDArray[np.float64]:
    """Tukey's weight function.

    Parameters
    ----------
    r : ndarray
        Residuals.
    c : float, optional
        Scale parameter. Default is 4.685.
    n_neg : int, optional
        Weight will be 1.0 for `n_neg` consecutive negative residuals.

    Returns
    -------
    w : ndarray
        Weight values.
    """
    u = r / c
    sqrt_w = np.maximum(1 - u * u, 0)
    n_neg = int(n_neg)
    if n_neg > 0:
        if n_neg == 1:
            mask = u < 0.0
        else:
            mask = mark_consecutive_negatives(u, n_neg)
        sqrt_w[mask] = 1.0
    return sqrt_w * sqrt_w


def median_absolute_value(values: NDArray[np.float64]) -> np.floating:
    """Median absolute value.

    Parameters
    ----------
    values : ndarray
        Values.

    Returns
    -------
    float
        Median absolute value.
    """
    return 1.4826 * np.median(np.abs(values))
