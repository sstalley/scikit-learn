"""Gaussian Mixture Model with Laplacian Regulization."""

# Author: Sean O. Stalley <sstalley@pdx.edu>
# License: TBD (probably BSD 3 clause)

import numpy as np
from ._base import _check_X
from ._gaussian_mixture import GaussianMixture
from ..utils import check_random_state
from ..utils.validation import _deprecate_positional_args

def _lap_reg(prob, laplacian):
    """calculate the laplacian regularizer

    parameters
    ----------
    prob      : array-like of shape (n_samples, n_component)
    laplacian : array-like of shape (n_samples, n_samples)

    returns
    -------
    lap_reg : array, shape (n_component) (?)
    """
    (n_samples, n_component) = prob.shape
    assert laplacian.shape == (n_samples, n_samples)

    # print("our regulizer was called!")

    lap_reg = np.empty((n_component))

    for k in range(n_component):
        fk = prob[:, k]
        lap_reg[k] = fk @ laplacian @ fk

    return lap_reg

def _lap_reg_2(prob, similarity):
    """calculate the laplacian regularizer

    parameters
    ----------
    prob      : array-like of shape (n_samples, n_component)
    laplacian : array-like of shape (n_samples, n_samples)

    returns
    -------
    prob      : array, shape (n_samples, n_component)
    """
    (n_samples, n_component) = prob.shape
    assert similarity.shape == (n_samples, n_samples)

    prob = similarity @ prob / np.sum(similarity, axis=0)

    return prob 


class LapRegGaussianMixture(GaussianMixture):

    @_deprecate_positional_args
    def __init__(self, n_components=1, *, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10,
                 laplacian=None, lap_mag=None, lap_reduce=0.9):
        super().__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval,
            covariance_type=covariance_type, weights_init=weights_init,
            means_init=means_init, precisions_init=precisions_init)

        # TODO SOS: make the laplacian an arugment passed at runtime
        self.laplacian = laplacian # Laplacian used for regularization
        self.lap_mag = lap_mag # Scaling factor for regularization
        self.lap_reduce = lap_reduce # controls how quickly the smoothing is reduced
        self.lap_smooth = self.lap_reduce # holds the current smoothing factor

        # we use this a lot, so save it
        similarity = -1 * laplacian
        similarity.setdiag(0)
        similarity.eliminate_zeros()

        self.similarity = similarity

        # Hacky lower bound shadow - should pass lower bound or restructure fit_predict
        self.lower_bound_HAX = -np.infty

    # Overloading this function from the 'base' model to add smoothing
    def _e_step(self, X):
        """E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        # HACK: shadow the lower-bound so we know what it is without 
        lower_bound = self.lower_bound_HAX

        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)

        while True:

            #smooth here
            reg2 = _lap_reg_2(np.exp(log_resp), self.similarity)
            log_resp = (1 - self.lap_smooth) * log_resp + self.lap_smooth * reg2

            # update the log_prob_norm (since we updated the probabilities)
            log_prob_norm, _ = self._estimate_log_prob_resp(X)
        
            # HACK - we call this here so we can make sure our values are geud
            # This will be immediately be called again when this function returns
            self._m_step(X, log_resp)

            lower_bound = self._compute_lower_bound(log_resp, np.mean(log_prob_norm))

            change = lower_bound - prev_lower_bound
            if change >= 0:
                break

            # If we didn't get better, apply more smoothing and try again
            self.lap_smooth = self.lap_smooth * self.lap_reduce
            print("lap_smooth %.5f ll change %.5f" % (self.lap_smooth, change))

        self.lower_bound_HAX = lower_bound

        return np.mean(log_prob_norm), log_resp

    def _compute_lower_bound(self, log_resp, log_prob_norm):
        return log_prob_norm - self.lap_mag * _lap_reg(np.exp(log_resp), self.laplacian)
