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
            data,
            kernel: Kernel,
            weighting_function,
            mean_function: Optional = None,
            noise_variance: Optional = None,
            likelihood: Optional[Gaussian] = None,
            Sigma_f_structure: Optional[str]="c_f",
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
        _, Y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data_input_to_tensor(data)
        self.weighting_function = weighting_function
        self.alpha = Parameter(0.5, transform=positive(), dtype=default_float())
        if Sigma_f_structure=="c_f":
            self.c_f= Parameter(1., transform=positive(), dtype=default_float())
        else:
            #I sample some points at random
            self.X_u=Parameter()
        self.prop=prop
        self.method=method

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


    def predictive_density(self,weighted=False) -> tf.Tensor:
        """Returns the training loss."""
        X, y = self.data
        c = self.maximize_c(y, self.prop)
        W = w.IMQ(c)
        self.w_full = tf.cast(W.W(X, y), dtype=default_float())
        self.m_w_full = W.dy(X, y) ** 2
        
        X_train, Y_train, X_test, Y_test, train_indices, test_indices = self.divide_batch(X, y)
        K = self.kernel(X_train)
        w_train = tf.gather(self.w_full, train_indices)
        m_w = tf.gather(self.m_w_full, train_indices)
        J_w = tf.linalg.tensor_diag(tf.squeeze(tf.math.pow(w_train, -2)))
        J_w_inv = tf.linalg.tensor_diag(tf.squeeze(tf.math.pow(w_train, 2)))
        if self.Sigma_f_structure=="c_f":
            Sigma_f = (self.c_f * K @ tf.linalg.inv(K + self.likelihood.variance * J_w) * self.likelihood.variance / 2) @ J_w
        if self.Sigma_f_structure=="Inducing+J_w":
            K_uu = self.kernel.matrix(self.X_u, self.X_u)
            K_ux_train = self.kernel.matrix(self.X_u, X)
            Sigma_f =  tf.linalg.inv(tf.linalg.inv(tf.transpose(K_ux_train)@tf.linalg.inv(K_uu) @ K_ux_train)+J_w_inv/self.likelihood.variance)
        else:
            K_uu = self.kernel.matrix(self.X_u, self.X_u)
            K_ux_train = self.kernel.matrix(self.X_u, X)
            Sigma_f = tf.transpose(K_ux_train)@tf.linalg.inv(K_uu) @ K_ux_train
        Sigma_a = self.alpha * K + (1 - self.alpha) * Sigma_f
        Sigma_a_inv = tf.linalg.inv(Sigma_a)
        mu_f = self.update_mu_f(self.alpha, Sigma_a_inv, J_w_inv, self.likelihood.variance, Y_train, m_w)

        K_star = tf.transpose(self.kernel.matrix(X_test, X_train))
        K_star_star = self.kernel.matrix(X_test, X_test)
        mu_f_pred = tf.transpose(K_star) @ tf.linalg.inv(Sigma_f) @ mu_f
        sigma_f_pred = K_star_star-tf.transpose(K_star)@tf.linalg.inv(Sigma_f)@K_star
        if weighted:
            sigma_f_pred=sigma_f_pred*tf.gather(self.w_full, test_indices) ** (2)
        sigma_f_pred = tf.linalg.diag_part(sigma_f_pred)  # *tf.gather(w_full, test_indices)**(-2)
        vv = tfp.distributions.Normal(mu_f_pred, sigma_f_pred)
        return -vv.log_prob(Y_test)



    @tf.function
    def update_mu_f(self, alpha: tf.Tensor, Sigma_a_inv: tf.Tensor, J_w_inv: tf.Tensor, 
                    sigma_2: tf.Tensor, Y_train: tf.Tensor, mu_w: tf.Tensor) -> tf.Tensor:
        """Updates the mean function parameters."""
        q1 = tf.linalg.inv(alpha * Sigma_a_inv + 2 * J_w_inv / sigma_2)
        q2 = tf.matmul((2 * J_w_inv / sigma_2), (Y_train - mu_w))
        return tf.matmul(q1, q2)

    def divide_batch(self, X: tf.Tensor, y: tf.Tensor, test_ratio: float = 0.2, 
                    seed: Optional[int] = None) -> Tuple[tf.Tensor, ...]:
        """Divides data into training and test batches."""
        if seed is None:
            seed = np.random.uniform(0, 10000)
        random.seed(seed)
        n = tf.shape(X)[0]
        num_test = tf.cast(tf.math.round(test_ratio * tf.cast(n, tf.float32)), tf.int32)
        indices = tf.range(n)
        shuffled_indices = tf.random.shuffle(indices)
        test_indices = shuffled_indices[:num_test]
        train_indices = shuffled_indices[num_test:]
        return (tf.gather(X, train_indices), tf.gather(y, train_indices),
                tf.gather(X, test_indices), tf.gather(y, test_indices),
                train_indices, test_indices)

    def maximize_c(self, y: tf.Tensor, quant: float) -> float:
        """Computes the maximum value of c based on quantile."""
        return np.quantile(abs(y), 1 - quant)

    def full_sample_estimate(self, X: tf.Tensor, Y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Computes the full sample estimate of mean and covariance."""
        K = self.kernel(X)
        self.w_full = self.likelihood.variance / (2 ** 0.5) * self.w_full
        J_w = tf.linalg.tensor_diag(tf.squeeze(tf.math.pow(self.w_full, -2))) * self.likelihood.variance ** 2 / 2
        J_w_inv = tf.linalg.tensor_diag(tf.squeeze(tf.math.pow(self.w_full, 2))) * 2 / (self.likelihood.variance ** 2)
        if self.Sigma_f_structure=="c_f":
            Sigma_f = (self.c_f * K @ tf.linalg.inv(K + self.likelihood.variance * J_w) * self.likelihood.variance / 2) @ J_w
        if self.Sigma_f_structure=="Inducing+J_w":
            K_uu = self.kernel.matrix(self.X_u, self.X_u)
            K_ux_train = self.kernel.matrix(self.X_u, X)
            Sigma_f =  tf.linalg.inv(tf.linalg.inv(tf.transpose(K_ux_train)@tf.linalg.inv(K_uu) @ K_ux_train)+J_w_inv/self.likelihood.variance)
        else:
            K_uu = self.kernel.matrix(self.X_u, self.X_u)
            K_ux_train = self.kernel.matrix(self.X_u, X)
            Sigma_f = tf.transpose(K_ux_train)@tf.linalg.inv(K_uu) @ K_ux_train
        Sigma_a = self.alpha * K + (1 - self.alpha) * self.Sigma_f
        Sigma_a_inv = tf.linalg.inv(Sigma_a)
        self.mu_f = self.update_mu_f(self.alpha, Sigma_a_inv, J_w_inv, self.likelihood.variance ** 2,
                               Y, tf.zeros_like(Y))
        


    def training_GVI(self):
        X, y = self.data
        c = self.maximize_c(y, self.prop)
        W = w.IMQ(c)
        self.w_full = tf.cast(W.W(X, y), dtype=default_float())
        self.m_w_full = W.dy(X, y) ** 2
        J_w = tf.linalg.tensor_diag(tf.squeeze(tf.math.pow(self.w_full, -2)))
        J_w_inv = tf.linalg.tensor_diag(tf.squeeze(tf.math.pow(self.w_full, 2)))
        m_w = self.m_w_full
        K = self.kernel(X)
        if self.Sigma_f_structure=="c_f":
            Sigma_f = (self.c_f * K @ tf.linalg.inv(K + self.likelihood.variance * J_w) * self.likelihood.variance / 2) @ J_w
        if self.Sigma_f_structure=="Inducing+J_w":
            K_uu = self.kernel.matrix(self.X_u, self.X_u)
            K_ux_train = self.kernel.matrix(self.X_u, X)
            Sigma_f =  tf.linalg.inv(tf.linalg.inv(tf.transpose(K_ux_train)@tf.linalg.inv(K_uu) @ K_ux_train)+J_w_inv/self.likelihood.variance)
        else:
            K_uu = self.kernel.matrix(self.X_u, self.X_u)
            K_ux_train = self.kernel.matrix(self.X_u, X)
            Sigma_f = tf.transpose(K_ux_train)@tf.linalg.inv(K_uu) @ K_ux_train
        Sigma_a = self.alpha * K + (1 - self.alpha) * Sigma_f
        Sigma_a_inv = tf.linalg.inv(Sigma_a)

        mu_f = self.update_mu_f(self.alpha, Sigma_a_inv, J_w_inv, self.likelihood.variance ** 2, y, m_w)
        loss = 0.5 * self.likelihood.variance ** (-2) * tf.transpose(mu_f) @ J_w_inv @ mu_f - self.likelihood.variance** (-2) * tf.transpose(
            mu_f) @ J_w_inv @ (y - m_w) + self.alpha / 2 * tf.transpose(
            mu_f) @ Sigma_a_inv @ mu_f  # -1/(2*(alpha-1))*tf.math.log(tf.linalg.det(Sigma_a)*tf.linalg.det(Sigma_f)**(alpha-1)*tf.linalg.det(K)**(-alpha))
        sigma = tf.transpose(y - mu_f - m_w) @ J_w_inv @ (y - mu_f - m_w) / tf.reduce_sum(J_w_inv)
        self.sigma = tf.sqrt(sigma)
        return loss

    def predict_f(self, X_new: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        pass
