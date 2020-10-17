from typing import Tuple

import numpy as np


def gaussian_mixture_moments(
    w: np.ndarray,  # the mixture weights shape=(N,)
    mean: np.ndarray,  # the mixture means shape(N, n)
    cov: np.ndarray,  # the mixture covariances shape (N, n, n)
) -> Tuple[
    np.ndarray, np.ndarray
]:  # the mean and covariance of the mixture shapes ((n,), (n, n))
    """Calculate the first two moments of a Gaussian mixture"""

    # mean
    mean_bar = w.dot(mean)  # TODO: hint np.average using axis and weights argument

    # covariance
    # # internal covariance
    cov_int = 0# TODO: hint, also an average
    for i in range(0, len(w)):
        cov_int = cov_int + w[i]*cov[i]

    # # spread of means
    # Optional calc: 
    sum_term = 0
    for i in range(0, len(w)):
        sum_term = sum_term + w[i]*mean[i]*mean[i]
    mean_diff = mean_bar.dot(np.transpose(mean_bar))
    cov_ext = sum_term - mean_diff  # TODO: hint, also an average

    # # total covariance
    cov_bar = cov_int + cov_ext  # TODO

    return mean_bar, cov_bar
