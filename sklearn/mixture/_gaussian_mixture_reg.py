"""Gaussian Mixture Model with Laplacian Regulization."""

# Author: Sean O. Stalley <sstalley@pdx.edu>
# License: TBD (probably BSD 3 clause)

import numpy as np
from scipy import sparse
from ._base import _check_X
from ._gaussian_mixture import GaussianMixture
from ..utils import check_random_state
from ..utils.validation import _deprecate_positional_args
from ..cluster import SpectralClustering

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

    return np.sum(lap_reg)

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

    prob = np.asarray(similarity @ prob / np.sum(similarity, axis=1))

    assert prob.shape == (n_samples, n_component)
    return prob 

class LapRegGaussianMixture(GaussianMixture):

    @_deprecate_positional_args
    def __init__(self, n_components=1, *, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10,
                 laplacian=None, lap_mag=None, lap_reduce=0.9, lap_tol=1e-5,
                 lap_reg=None):
        super().__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval,
            covariance_type=covariance_type, weights_init=weights_init,
            means_init=means_init, precisions_init=precisions_init)

        # TODO SOS: make the laplacian an arugment passed at runtime
        if lap_reg is not None:
            dinv = np.reciprocal(laplacian.diagonal())
            if lap_reg == "rw": # Random Walk
                print("random walk regularizing laplacian...")
                laplacian = np.diag(dinv) @ laplacian
            elif lap_reg == "sym": # Symmetric
                print("symmetric regularizing laplacian...")
                Dsqrt = np.diag(np.sqrt(dinv))
                laplacian = Dsqrt @ laplacian @ Dsqrt

        laplacian = sparse.csr_matrix(laplacian)


        self.laplacian = laplacian # Laplacian used for regularization
        self.lap_mag = lap_mag # Scaling factor for regularization
        self.lap_reduce = lap_reduce # controls how quickly the smoothing is reduced
        self.lap_smooth = lap_reduce # holds the current smoothing factor
        self.lap_tol = lap_tol # how low we allow the smoothing factor to get before giving up

        # we use this a lot, so save it
        similarity = -1 * laplacian
        similarity.setdiag(0)
        similarity.eliminate_zeros()

        self.similarity = similarity

    # Overloading to add spectral clustering initialization
    def _initialize_parameters(self, X, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        n_samples, _ = X.shape

        if self.init_params == 'spectral':
            print("initializing using spectral clustering...")
            resp = np.zeros((n_samples, self.n_components))
            label = SpectralClustering(n_clusters=self.n_components, affinity='precomputed',
                                   random_state=random_state).fit(self.similarity).labels_
            resp[np.arange(n_samples), label] = 1
            print(np.argmax(resp, axis=1))
            self._initialize(X, resp)
        else:
            super()._initialize_parameters(X, random_state)



    # Overloading this function from the 'base' model to add smoothing
    def fit_predict(self, X, y=None):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        .. versionadded:: 0.20

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        X = _check_X(X, self.n_components, ensure_min_samples=2)
        self._check_n_features(X, reset=True)
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not(self.warm_start and hasattr(self, 'converged_'))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)

            lower_bound = (-np.infty if do_init else self.lower_bound_)

            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound

                #log_prob_norm, log_resp = self._e_step(X)
                log_prob_norm_orig, log_resp_orig = self._estimate_log_prob_resp(X)
                while self.lap_smooth > self.lap_tol:

                    #smooth here
                    reg2 = _lap_reg_2(np.exp(log_resp_orig), self.similarity)
                    log_resp = (1 - self.lap_smooth) * log_resp_orig + self.lap_smooth * reg2
                    assert (n_samples, self.n_components) == log_resp.shape
        
                    # update the log_prob_norm (since we updated the probabilities)
                    log_prob_norm, _ = self._estimate_log_prob_resp(X)
                
                    # print("log_resp.shape", log_resp.shape)
                    # print("X.shape", X.shape)
        
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

                # If we couldn't make it better with smoothing, return what we had before
                if self.lap_smooth <= self.lap_tol:
                    log_prob_norm = log_prob_norm_orig
                    log_resp = log_resp_orig
                    lower_bound = prev_lower_bound
        
                # self._m_step(X, log_resp)
                # lower_bound = self._compute_lower_bound(
                #     log_resp, log_prob_norm)

                change = lower_bound - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)

                if change < self.tol:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end(lower_bound)

            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = n_iter

        if not self.converged_:
            warnings.warn('Initialization %d did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.'
                          % (init + 1), ConvergenceWarning)

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._estimate_log_prob_resp(X)

        return log_resp.argmax(axis=1)

    def _e_step(self, X):
        # This shouldn't ever get used anymore, but if it does we want to know
        # about it and scream as loudly as we can
        assert False

    def _compute_lower_bound(self, log_resp, log_prob_norm):
        return log_prob_norm - self.lap_mag * _lap_reg(np.exp(log_resp), self.laplacian)
