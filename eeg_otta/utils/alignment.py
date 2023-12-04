from typing import Optional

import numpy as np
from pyriemann.utils.mean import mean_riemann
from scipy import linalg


def _euclidean_alignment(x: np.ndarray, x_test: np.ndarray):
    R = np.matmul(x, x.transpose((0, 2, 1))).mean(0)
    R_op = linalg.inv(linalg.sqrtm(R))
    x = np.matmul(R_op, x)
    x_test = np.matmul(R_op, x_test)
    return x, x_test


def _riemannian_alignment(x: np.ndarray, x_test: np.ndarray):
    covmats = np.matmul(x, x.transpose((0, 2, 1)))
    R = mean_riemann(covmats)
    R_op = linalg.inv(linalg.sqrtm(R))
    x = np.matmul(R_op, x)
    x_test = np.matmul(R_op, x_test)
    return x, x_test


def _align(method: str | bool | None, x: np.ndarray, x_test: np.ndarray,
                 train_domains: np.ndarray, test_domains: np.ndarray):
    if train_domains is None:  # only one domain
        train_domains, test_domains = np.zeros(x.shape[0]), np.zeros(x_test.shape[0])
    for domain in np.unique(train_domains):
        if method == "euclidean":
            x[train_domains == domain], x_test[
                test_domains == domain] = _euclidean_alignment(
                x[train_domains == domain], x_test[test_domains == domain])
        elif method == "riemann":
            x[train_domains == domain], x_test[
                test_domains == domain] = _riemannian_alignment(
                x[train_domains == domain], x_test[test_domains == domain])
        elif method in [False, None]:
            pass
        else:
            raise NotImplementedError
    return x, x_test


def align(method: str | bool | None, x: np.ndarray, x_test: Optional[np.ndarray] = None,
          train_domains: Optional[np.ndarray] = None,
          test_domains: Optional[np.ndarray] = None):
    return _align(method, x, x_test, train_domains=train_domains,
                  test_domains=test_domains)
