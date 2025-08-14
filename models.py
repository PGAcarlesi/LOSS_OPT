import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import random
from gpflow.base import Module, Parameter
from gpflow.kernels import Kernel
from gpflow.likelihoods import Gaussian
from gpflow.models.model import GPModel
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.config import default_float
from gpflow.utilities import positive, triangular
from typing import Optional, Tuple, Union
from check_shapes import check_shapes, inherit_check_shapes

# Import the IMQ functionality
try:
    import w  # Assuming w.py contains the IMQ class
except ImportError:
    class IMQ:
        def __init__(self, c):
            self.c = c
            
        def W(self, X, y):
            return tf.ones_like(y)
            
        def dy(self, X, y):
            return tf.zeros_like(y)
    
    w = type('w', (), {'IMQ': IMQ})()

class StandardGPR(GPModel, InternalDataTrainingLossMixin):
    """
    Standard Gaussian Process Regression model.
    
    This model implements the standard GP regression with a Gaussian likelihood.
    
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
    sigma : float
        Initial value for the noise variance
    """
    
    @check_shapes(
        "data[0]: [N, D]",
        "data[1]: [N, P]",
    )
    def __init__(
        self,
        data: Tuple[tf.Tensor, tf.Tensor],
        kernel: Optional[Kernel] = None,
        likelihood: Optional[Gaussian] = None,
        variance: float = 1.0,
        lengthscale: float = 1.0,
        sigma: float = 0.01,
    ):
        if kernel is None:
            kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(variance, lengthscale)
        if likelihood is None:
            likelihood = Gaussian(sigma)
            
        super().__init__(kernel, likelihood)
        self.data = data

    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        """Returns the objective to be maximized."""
        return self.training_loss()

    @check_shapes(
        "return: []",
    )
    def training_loss(self) -> tf.Tensor:
        """Returns the training loss."""
        X, y = self.data
        K = self.kernel(X)
        return -self.log_marginal_likelihood()

    def predict_f(self, Xnew: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict the mean and variance of the latent function at new points.
        
        Parameters:
        -----------
        Xnew : tf.Tensor
            New input points
            
        Returns:
        --------
        Tuple[tf.Tensor, tf.Tensor]
            Mean and variance of predictions
        """
        X, y = self.data
        K = self.kernel(X)
        K_s = self.kernel(X, Xnew)
        K_ss = self.kernel(Xnew)
        
        # Compute mean
        L = tf.linalg.cholesky(K + self.likelihood.variance * tf.eye(tf.shape(X)[0], dtype=K.dtype))
        alpha = tf.linalg.triangular_solve(tf.transpose(L), 
                                        tf.linalg.triangular_solve(L, y))
        mean = tf.matmul(K_s, alpha, transpose_a=True)
        
        # Compute variance
        v = tf.linalg.triangular_solve(L, K_s)
        var = K_ss - tf.matmul(v, v, transpose_a=True)
        
        return mean, var

    def predict_y(self, Xnew: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict the mean and variance of the observations at new points.
        
        Parameters:
        -----------
        Xnew : tf.Tensor
            New input points
            
        Returns:
        --------
        Tuple[tf.Tensor, tf.Tensor]
            Mean and variance of predictions
        """
        mean, var = self.predict_f(Xnew)
        return mean, var + self.likelihood.variance 

class GaussianProcessModel(GPModel, InternalDataTrainingLossMixin):
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
    
    @check_shapes(
        "data[0]: [N, D]",
        "data[1]: [N, P]",
    )
    def __init__(
        self,
        data: Tuple[tf.Tensor, tf.Tensor],
        kernel: Optional[Kernel] = None,
        likelihood: Optional[Gaussian] = None,
        variance: float = 4.5,
        lengthscale: float = 0.9,
        alpha: float = 0.5,
        sigma: float = 0.01,
        c_f: float = 1.0,
        prop: float = 0.1,
    ):
        if kernel is None:
            kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(variance, lengthscale)
        if likelihood is None:
            likelihood = Gaussian(sigma)
            
        super().__init__(kernel, likelihood)
        self.data = data
        self.alpha = Parameter(alpha, transform=positive(), dtype=default_float())
        self.c_f = Parameter(c_f, transform=positive(), dtype=default_float())
        self.prop = prop

    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        """Returns the objective to be maximized."""
        return self.training_loss()

    @check_shapes(
        "return: []",
    )
    def training_loss(self) -> tf.Tensor:
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
        
        Sigma_f = (self.c_f * K @ tf.linalg.inv(K + self.likelihood.variance * J_w) * self.likelihood.variance / 2) @ J_w
        Sigma_a = self.alpha * K + (1 - self.alpha) * Sigma_f
        Sigma_a_inv = tf.linalg.inv(Sigma_a)
        mu_f = self.update_mu_f(self.alpha, Sigma_a_inv, J_w_inv, self.likelihood.variance, Y_train, m_w)
        
        return -tf.reduce_sum(tfp.distributions.Normal(mu_f, tf.linalg.diag_part(Sigma_f)).log_prob(Y_test))

    def update_mu_f(self, alpha: tf.Tensor, Sigma_a_inv: tf.Tensor, J_w_inv: tf.Tensor, 
                    sigma_2: tf.Tensor, Y_train: tf.Tensor, mu_w: tf.Tensor) -> tf.Tensor:
        """Updates the mean function parameters."""
        q1 = tf.linalg.inv(alpha * Sigma_a_inv + 2 * J_w_inv / sigma_2)
        q2 = tf.matmul((2 * J_w_inv / sigma_2), (Y_train - mu_w))
        return tf.matmul(q1, q2)

    def update_Sigma_f(self, c_f: tf.Tensor, K: tf.Tensor, sigma_2: tf.Tensor, 
                      J_w: tf.Tensor) -> tf.Tensor:
        """Updates the covariance function parameters."""
        return c_f * K @ tf.linalg.inv(K + sigma_2 * J_w / 2) * sigma_2 * J_w / 2

    def divide_batch(self, X: tf.Tensor, y: tf.Tensor, test_ratio: float = 0.2, 
                    seed: Optional[int] = None) -> Tuple[tf.Tensor, ...]:
        """Divides data into training and test batches."""
        if seed is None:
            seed = np.random.uniform(0, 1000)
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
        
        Sigma_f = (self.c_f * K @ tf.linalg.inv(K + self.likelihood.variance ** 2 / 2 * J_w) * 
                  self.likelihood.variance ** 2 / 2) @ J_w
        Sigma_a = self.alpha * K + (1 - self.alpha) * Sigma_f
        Sigma_a_inv = tf.linalg.inv(Sigma_a)
        mu_f = self.update_mu_f(self.alpha, Sigma_a_inv, J_w_inv, self.likelihood.variance ** 2, 
                               Y, tf.zeros_like(Y))
        return mu_f, Sigma_f

    def predict_f(self, Xnew: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict the mean and variance of the latent function at new points.
        
        Parameters:
        -----------
        Xnew : tf.Tensor
            New input points
            
        Returns:
        --------
        Tuple[tf.Tensor, tf.Tensor]
            Mean and variance of predictions
        """
        X, y = self.data
        c = self.maximize_c(y, self.prop)
        W = w.IMQ(c)
        self.w_full = tf.cast(W.W(X, y), dtype=default_float())
        
        K = self.kernel(X)
        K_s = self.kernel(X, Xnew)
        K_ss = self.kernel(Xnew)
        
        J_w = tf.linalg.tensor_diag(tf.squeeze(tf.math.pow(self.w_full, -2)))
        J_w_inv = tf.linalg.tensor_diag(tf.squeeze(tf.math.pow(self.w_full, 2)))
        
        Sigma_f = self.update_Sigma_f(self.c_f, K, self.likelihood.variance, J_w)
        Sigma_a = self.alpha * K + (1 - self.alpha) * Sigma_f
        Sigma_a_inv = tf.linalg.inv(Sigma_a)
        
        mu_f = self.update_mu_f(self.alpha, Sigma_a_inv, J_w_inv, self.likelihood.variance, y, tf.zeros_like(y))
        
        # Compute predictive mean and variance
        v = tf.linalg.triangular_solve(tf.linalg.cholesky(K + self.likelihood.variance * J_w), K_s)
        mean = tf.matmul(K_s, tf.linalg.solve(K + self.likelihood.variance * J_w, y))
        var = K_ss - tf.matmul(v, v, transpose_a=True)
        
        return mean, var

    def predict_y(self, Xnew: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict the mean and variance of the observations at new points.
        
        Parameters:
        -----------
        Xnew : tf.Tensor
            New input points
            
        Returns:
        --------
        Tuple[tf.Tensor, tf.Tensor]
            Mean and variance of predictions
        """
        mean, var = self.predict_f(Xnew)
        return mean, var + self.likelihood.variance 

class GVI_SVGP_alpha(GPModel, InternalDataTrainingLossMixin):
    """
    Sparse Variational Gaussian Process model with alpha parameter.
    
    This model implements a sparse variational GP regression with an alpha parameter
    that controls the balance between the prior and variational distributions.
    
    Parameters:
    -----------
    inducing_variable : tf.Tensor
        The inducing points
    N_u : int
        Number of inducing points
    alpha : float
        Balance parameter between prior and variational distributions
    """
    
    def __init__(
        self,
        inducing_variable: tf.Tensor,
        N_u: int,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.inducing_variable = inducing_variable
        self.N_u = N_u

        # Variational and GP parameters
        self.alpha = Parameter(alpha, trainable=False)
        self.mu_q = Parameter(tf.ones((N_u, 1)))
        self.var_q_L = Parameter(np.eye(N_u), transform=triangular())
        self.variance = Parameter(1.0, transform=positive())
        self.lengthscale = Parameter(1.0, transform=positive())
        self.sigma = Parameter(1.0, transform=positive())

    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        """Returns the objective to be maximized."""
        return self.training_loss()

    @check_shapes(
        "X: [N, D]",
        "y: [N, P]",
        "return: []",
    )
    def train_step(self, X: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Compute the training loss for a single step.
        
        Parameters:
        -----------
        X : tf.Tensor
            Input features
        y : tf.Tensor
            Target values
            
        Returns:
        --------
        tf.Tensor
            The training loss
        """
        N = tf.shape(X)[0]  # Get batch size dynamically
        
        # Build kernel matrices
        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(self.variance, self.lengthscale)
        K_uu = kernel.matrix(self.inducing_variable, self.inducing_variable)
        K_uf = kernel.matrix(self.inducing_variable, X)
        K_fu = tf.transpose(K_uf)

        # Invert via stable_solve
        jitter = tf.eye(self.N_u, dtype=default_float()) * 1e-6
        K_uu_inv = tf.linalg.inv(K_uu + jitter)

        # Variational covariances
        Sigma_u = self.var_q_L @ tf.transpose(self.var_q_L)
        Sigma_a = self.alpha * K_uu + (1.0 - self.alpha) * Sigma_u
        Sigma_a_inv = tf.linalg.inv(Sigma_a + jitter)

        # Log-determinants
        ld_K = tf.linalg.logdet(K_uu + jitter)
        ld_Sig_a = tf.linalg.logdet(Sigma_a + jitter)
        ld_Sig_u = tf.linalg.logdet(Sigma_u + jitter)

        # Loss components
        residual = y - K_fu @ K_uu_inv @ self.mu_q
        term1 = 0.5 * tf.reduce_sum(tf.square(residual)) * (self.sigma**-2)
        term2 = 0.5 * self.alpha * tf.reduce_sum(self.mu_q * (Sigma_a_inv @ self.mu_q))
        term3 = N * tf.math.log(self.sigma)
        T1 = K_uu_inv @ (K_uf @ K_fu)
        T2 = T1 @ (K_uu_inv @ Sigma_u)
        term4 = 0.5 * (self.sigma**-2) * (tf.reduce_sum(kernel.apply(X, X)) - tf.linalg.trace(T1) + tf.linalg.trace(T2))
        term5 = 1.0/(2.0*(1.0-self.alpha)) * (ld_Sig_a - (1-self.alpha)*ld_Sig_u - self.alpha*ld_K)
        
        loss = term1 + term2 + term3 + term4 + term5
        return tf.squeeze(loss)

    @check_shapes(
        "Xnew: [N, D]",
        "return: [N, P]",
    )
    def predict_f(self, Xnew: tf.Tensor) -> tf.Tensor:
        """
        Predict the mean of the latent function at new points.
        
        Parameters:
        -----------
        Xnew : tf.Tensor
            New input points
            
        Returns:
        --------
        tf.Tensor
            Mean of predictions
        """
        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(self.variance, self.lengthscale)
        K_uu = kernel.matrix(self.inducing_variable, self.inducing_variable)
        K_xu = kernel.matrix(Xnew, self.inducing_variable)
        I_M = tf.eye(self.N_u, dtype=default_float())

        K_uu_inv = tf.linalg.inv(K_uu + I_M * 1e-6)
        return K_xu @ K_uu_inv @ self.mu_q

    @check_shapes(
        "Xnew: [N, D]",
        "return: [N, P]",
    )
    def predict_y(self, Xnew: tf.Tensor) -> tf.Tensor:
        """
        Predict the mean of the observations at new points.
        
        Parameters:
        -----------
        Xnew : tf.Tensor
            New input points
            
        Returns:
        --------
        tf.Tensor
            Mean of predictions
        """
        return self.predict_f(Xnew) 