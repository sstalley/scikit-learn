"""Gaussian Mixture Model with Laplacian Regulization."""

# Author: Sean O. Stalley <sstalley@pdx.edu>
# License: TBD (probably BSD 3 clause)

import numpy as np
from ._gaussian_mixture import GaussianMixture
from ..utils.validation import _deprecate_positional_args

def _lap_reg(prob, laplacian):
    """Calculate the Laplacian Regularizer

    Parameters
    ----------
    prob      : array-like of shape (n_samples, n_component)
    laplacian : array-like of shape (n_samples, n_samples)

    Returns
    -------
    lap_reg : array, shape (n_component) (?)
    """
    (n_samples, n_component) = prob.shape
    assert laplacian.shape == (n_samples, n_samples)

    print("Our regulizer was called!")

    lap_reg = np.empty((n_component))

    for k in range(n_component):
        fk = prob[:, k]
        lap_reg[k] = np.dot(np.dot(fk, laplacian), fk)

    return lap_reg

class LapRegGaussianMixture(GaussianMixture):

    @_deprecate_positional_args
    def __init__(self, n_components=1, *, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10,
                 laplacian=None, lap_mag=None):
        super().__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval,
            covariance_type=covariance_type, weights_init=weights_init,
            means_init=means_init, precisions_init=precisions_init)

        self.laplacian = laplacian
        self.lap_mag = lap_mag

    # Overloading this function from the "base" mixture model to add our regularizer
    def _estimate_weighted_log_prob(self, X):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """

        print("Our overloaded function was called!")
        res = super()._estimate_weighted_log_prob(X)
        
        if self.laplacian is not None and self.lap_mag is not None:
            prob = np.exp(self._estimate_log_prob(X))
            res = res - self.lap_mag * _lap_reg(prob, self.laplacian)

        return res

