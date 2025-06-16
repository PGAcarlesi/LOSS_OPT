import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import random

from tensorflow.lite.python.schema_py_generated import TensorT

import w  # Assuming `w` is a module providing IMQ functionality
import gpflow
from gpflow.base import Module, Parameter
from gpflow.kernels import Kernel
from gpflow.likelihoods import Gaussian
from gpflow.models.model import GPModel
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.config import default_float
from gpflow.utilities import positive
from typing import Optional, Tuple, Union
from check_shapes import check_shapes, inherit_check_shapes
from gpflow.models.util import data_input_to_tensor


class Robust_gaussian_base_class(GPModel, InternalDataTrainingLossMixin):
    """
    A robust Gaussian Process model with outlier detection.

    This model implements a robust GP regression that can handle outliers by using
    a weighting function to downweight potentially problematic data points.

    Parameters:
    -----------
    data : Tuple[tf.Tensor, tf.Tensor]
        Training data (X, y)
    kernel : Optional[Kernel]
        The kernel to use. If None, uses RBF kernel
    likelihood : Optional[Gaussian]
        The likelihood to use. If None, uses Gaussian likelihood
    variance : float
        Initial value for the kernel variance
    lengthscale : float
        Initial value for the kernel lengthscale
    alpha : float
        Initial value for the alpha parameter
    sigma : float
        Initial value for the noise variance
    c_f : float
        Initial value for the c_f parameter
    prop : float
        Proportion of expected outliers
    """

    def __init__(
            self,
            X,
            y,
            kernel: Kernel,
            weighting_function,
            mean_function: Optional = None,
            noise_variance: Optional = None,
            likelihood: Optional[Gaussian] = None,
            Sigma_f_structure: Optional[str] = "c_f",
            prop: Optional[float] = 0.1,
            method: Optional[str] = "CV",
    ):
        assert (noise_variance is None) or (
                likelihood is None
        ), "Cannot set both `noise_variance` and `likelihood`."
        if likelihood is None:
            if noise_variance is None:
                noise_variance = 1.0
            likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=1)
        self.X=X
        self.y=y
        self.weighting_function = weighting_function
        self.alpha = Parameter(0.5, transform=positive(), dtype=default_float())
        if Sigma_f_structure == "c_f":
            self.c_f = Parameter(1., transform=positive(), dtype=default_float())
        else:
            # I sample some points at random
            self.X_u = Parameter()
        print(10)
        self.prop = prop
        self.method = method

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        """Returns the objective to be maximized."""
        return self.training_GVI()
        """
        match self.method:
            case "GVI":
                self.sigma=1.
                return self.GVI()
            case "Expected_Loss":
                return self.expected_loss()
            case "Predictive_density":
                return self.predictive_density()
            case "Predictive_weighted_density":
                return self.predictive_density(weighted=True)
            case _:
                return f"method {self.method} not implemented"
        """

    def predictive_density(self, weighted=False) -> tf.Tensor:
       pass

    @tf.function
    def update_mu_f(self, alpha: tf.Tensor, Sigma_a_inv: tf.Tensor, J_w_inv: tf.Tensor,
                    sigma_2: tf.Tensor, Y_train: tf.Tensor, mu_w: tf.Tensor) -> tf.Tensor:
        """Updates the mean function parameters."""
        pass

    def divide_batch(self, X: tf.Tensor, y: tf.Tensor, test_ratio: float = 0.2,
                     seed: Optional[int] = None) -> Tuple[tf.Tensor, ...]:
        """Divides data into training and test batches."""
        pass

    def maximize_c(self, y: tf.Tensor, quant: float) -> float:
        """Computes the maximum value of c based on quantile."""
        return np.quantile(abs(y), 1 - quant)

    def full_sample_estimate(self, X: tf.Tensor, Y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Computes the full sample estimate of mean and covariance."""
        pass

    def training_GVI(self):
        return 10
        """K=self.kernel.matrix(self.X,self.X)
        print(10)
        return 0.5*tf.reduce_sum(tf.transpose(self.y)@tf.linalg.inv(K+self.likelihood.variance)@self.y)+0.5*tf.linalg.logdet(K+self.likelihood.variance)
"""
    def predict_f(self, X_new: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        pass
