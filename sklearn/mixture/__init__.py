"""
The :mod:`sklearn.mixture` module implements mixture modeling algorithms.
"""

from ._gaussian_mixture import GaussianMixture
from ._gaussian_mixture_reg import LapRegGaussianMixture
from ._bayesian_mixture import BayesianGaussianMixture


__all__ = ['GaussianMixture',
           'LapRegGaussianMixture',
           'BayesianGaussianMixture']
