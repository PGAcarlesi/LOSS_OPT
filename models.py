from sklearn.model_selection import train_test_split
import time
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np



class GVI_GP_alpha(gpflow.models.BayesianModel):
    def __init__(self, X, y, Kernel, objective="GVI", loss_type="MSE", Sigma_f_structure="c_f",
                 lambda_method="constant", c_f_calibrating=False, inducing=[.0], lambda_pred_lev=0.95, alpha=0.5,
                 sigma=.25):
        super().__init__()
        self.X = X
        self.y = y

        # define if we are gonna optimize through the GVI or predictive.
        self.objective = objective
        if objective == "GVI":
            self.alpha = gpflow.Parameter(alpha, trainable=False)
        else:
            self.alpha = gpflow.Parameter(alpha, trainable=False)
            self.test_size = 0.1

        if loss_type != "MSE":
            self.sigma = gpflow.Parameter(sigma, transform=gpflow.utilities.positive())
        # define the type of objective
        self.loss_type = loss_type

        # GP parameters
        self.kernel = Kernel
        self.inducing_variable = inducing

        # set structure for \Sigma_f
        if Sigma_f_structure == "c_f" and not c_f_calibrating:
            self.c_f = gpflow.Parameter(1., transform=gpflow.utilities.positive(), trainable=True)
        elif Sigma_f_structure == "c_f" and c_f_calibrating:
            self.c_f = gpflow.Parameter(1., transform=gpflow.utilities.positive(), trainable=False)
        else:
            self.Kernel_inducing = gpflow.kernels.SquaredExponential(lengthscales=0.1, variance=0.1)
        self.Sigma_f_structure = Sigma_f_structure
        self.c_f_calibrating = c_f_calibrating

        # set lambda
        if lambda_method == "constant":
            self.lambda_i = tf.constant(1., dtype=tf.float64)
        elif lambda_method == "predictive" and objective == "predictive":
            self.lambda_i = gpflow.Parameter(1., transform=gpflow.utilities.positive(), trainable=True,
                                             dtype=tf.float64)
        else:
            print("lambda method unknown")
        self.lambda_method = lambda_method

    tf.keras.backend.set_floatx('float64')

    # needed for the class
    def maximum_log_likelihood_objective(self):
        pass

    # update Sigma_f according to method.
    @tf.function
    def compute_sigma_f(self, X):
        if self.Sigma_f_structure == "c_f":
            K = self.kernel(X, X)
            eye = tf.eye(tf.shape(K)[0], dtype=K.dtype)
            if self.loss_type == "MSE":
                Sigma_f = self.c_f * K @ tf.linalg.inv(K + eye / self.lambda_i) * self.lambda_i
            else:
                Sigma_f = self.c_f * K @ tf.linalg.inv(
                    K + self.sigma ** 2 / self.lambda_i * eye) * self.sigma ** 2 / self.lambda_i
        else:
            K = self.kernel(X, X)
            K_uu = self.Kernel_inducing(self.inducing_variable, self.inducing_variable)
            K_uf = self.Kernel_inducing(self.inducing_variable, X)
            K_fu = tf.transpose(K_uf)
            K_uu_inv = tf.linalg.inv(K_uu + tf.linalg.eye(K_uu.shape[0], dtype=tf.float64) * gpflow.default_jitter())
            Sigma_f = K_fu @ K_uu_inv @ K_uf
        return K, Sigma_f

    @tf.function
    def train_step(self):
        if self.objective == "GVI":
            loss = self.GVI_step()
        else:
            loss = self.Pred_step()
        return loss

    def mse_loss(self, y, mu_f, sigma_f):
        loss = 0.5 * tf.transpose(y - mu_f) @ (y - mu_f) * (self.lambda_i) + 0.5 * sigma_f * self.lambda_i
        return loss

    def log_likelihood_loss(self, y, mu_f, sigma_f):
        loss = 0.5 * tf.transpose(y - mu_f) @ (y - mu_f) * (self.sigma ** (-2) * self.lambda_i)
        loss += len(y) * tf.math.log(self.sigma)
        loss += 0.5 * self.sigma ** (-2) * tf.reduce_sum(sigma_f * self.lambda_i)
        return loss

    def pred_mse(self, y, mu_f, sigma_f):
        loss = 0.5 * tf.transpose(y - mu_f) @ (y - mu_f) + 0.5 * tf.reduce_sum(sigma_f)
        return loss

    def pred_log_prob(self, y, mu_f, sigma_y):
        # ensure 1-D vectors and float64 (gpflow/TensorFlow prefer float64 in many setups)
        y = tf.reshape(tf.cast(y, tf.float64), [-1])
        mu_f = tf.reshape(tf.cast(mu_f, tf.float64), [-1])
        sigma_y = tf.cast(sigma_y, tf.float64)

        # If sigma_y is a full covariance matrix, take diagonal (marginal variances)
        if sigma_y.shape.ndims == 2 and sigma_y.shape[0] == sigma_y.shape[1]:
            var = tf.linalg.diag_part(sigma_y)
        else:
            var = tf.reshape(sigma_y, [-1])


        # quadratic term (scalar)
        term_quad = 0.5 * tf.reduce_sum((y - mu_f) ** 2 / var)

        # log-determinant / normalizer term (scalar)
        term_log = 0.5 * tf.reduce_sum(tf.math.log(var))

        # optional extra term you had: make sure to treat self.sigma numerically
        try:
            inv_sigma2 = tf.math.reciprocal(tf.square(tf.cast(self.sigma, tf.float64)))
        except Exception:
            inv_sigma2 = tf.constant(0.0, dtype=tf.float64)

        term3 = 0.5 * inv_sigma2 * tf.reduce_sum(var + tf.square(tf.cast(self.sigma, tf.float64)))

        return term_quad + term_log + term3

    def GVI_step(self):
        K, Sigma_f = self.compute_sigma_f(self.X)

        N = tf.cast(tf.shape(K)[0], tf.float64)
        eye = tf.eye(N, dtype=K.dtype)
        jitter = eye * 1e-6

        Sigma_a = self.alpha * K + (1.0 - self.alpha) * Sigma_f
        Sigma_a_inv = tf.linalg.inv(Sigma_a + jitter)
        if self.loss_type == "MSE":
            q1 = tf.linalg.inv(self.alpha * Sigma_a_inv + eye * self.lambda_i)
            q2 = tf.matmul(eye * self.lambda_i, self.y)
        else:
            q1 = tf.linalg.inv(self.alpha * Sigma_a_inv + eye / self.sigma ** 2 * self.lambda_i)
            q2 = tf.matmul(eye / self.sigma ** 2 * self.lambda_i, self.y)
        mu_f = tf.matmul(q1, q2)

        # Log-determinants
        ld_Sig_a = tf.linalg.logdet(Sigma_a + jitter)
        ld_Sig_f = tf.linalg.logdet(Sigma_f + jitter)
        ld_K = tf.linalg.logdet(K + jitter)

        # Loss terms
        if self.loss_type == "MSE":
            term1 = self.mse_loss(self.y, mu_f, tf.linalg.trace(Sigma_f))
        else:
            term1 = self.log_likelihood_loss(self.y, mu_f, tf.linalg.trace(Sigma_f))
        term2 = 0.5 * self.alpha * tf.transpose(mu_f) @ (Sigma_a_inv) @ (mu_f)
        term3 = -1.0 / (2.0 * (self.alpha - 1.0)) * (ld_Sig_a + (self.alpha - 1.0) * ld_Sig_f - self.alpha * ld_K)
        loss = tf.squeeze(term1 + term2 + term3)

        if self.c_f_calibrating:
            residuals = self.y - mu_f

            # 95% CI bounds based on predicted std dev
            std_p = tf.sqrt(tf.linalg.diag_part(Sigma_f)+self.sigma**2)
            within_95 = tf.abs(residuals) <= 1.96 * std_p

            # Proportion inside CI
            coverage_95 = tf.reduce_mean(tf.cast(within_95, tf.float64))

            self.c_f.assign(self.c_f + 0.5 * (0.95 - coverage_95))

        return loss

    def Pred_step(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=0)

        K, Sigma_f = self.compute_sigma_f(X_train)

        N = tf.cast(tf.shape(K)[0], tf.float64)

        eye = tf.eye(N, dtype=K.dtype)
        jitter = eye * 1e-6
        K_inv = tf.linalg.inv(K + jitter)

        Sigma_a = self.alpha * K + (1.0 - self.alpha) * Sigma_f
        Sigma_a_inv = tf.linalg.inv(Sigma_a + jitter)
        if self.loss_type == "MSE":
            q1 = tf.linalg.inv(self.alpha * Sigma_a_inv + eye * self.lambda_i)
            q2 = tf.matmul(eye * self.lambda_i, y_train)
        else:
            q1 = tf.linalg.inv(self.alpha * Sigma_a_inv + eye / self.sigma ** 2 * self.lambda_i)
            q2 = tf.matmul(eye / self.sigma ** 2 * self.lambda_i, y_train)
        mu_f = tf.matmul(q1, q2)

        K_star = self.kernel(X_test, X_train)
        K_star_star = self.kernel(X_test, X_test)

        mu_p = K_star @ K_inv @ mu_f
        A = K_star_star - K_star @ K_inv @ tf.transpose(K_star) \
            + K_star @ K_inv @ Sigma_f @ tf.transpose(K_star @ K_inv)

        # use diagonal of covariance directly
        sigma_p = tf.linalg.diag_part(A)  # no inverse here

        if self.loss_type == "MSE":
            term1 = self.pred_mse(y_test, mu_p, sigma_p)
        else:
            # additive observation noise is variance, not precision
            var_y = sigma_p + tf.cast(self.sigma ** 2, tf.float64)
            term1 = self.pred_log_prob(y_test, mu_p, var_y)

        if self.c_f_calibrating:
            residuals = y_test - mu_p

            # 95% CI bounds based on predicted std dev
            std_p = tf.sqrt(sigma_p)
            within_95 = tf.abs(residuals) <= 1.96 * std_p

            # Proportion inside CI
            coverage_95 = tf.reduce_mean(tf.cast(within_95, tf.float64))

            self.c_f.assign(self.c_f + 0.5 * (0.95 - coverage_95))
        return term1


    def sample_y(self, n_samples=100):
        N = self.X.shape[0]
        f = np.random.multivariate_normal(mean=tf.squeeze(self.mu_f), cov=self.Sigma_f, size=n_samples)  # put the
        if self.loss_type == "MSE":
            return f
        else:
            noise = tf.random.normal(
                shape=(n_samples, N),
                mean=0.0,
                stddev=self.sigma,  # can be Parameter/Variable
                dtype=tf.float64
            )
            return f + noise

    def predict_ins(self):
        K, Sigma_f = self.compute_sigma_f(self.X)

        N = tf.cast(tf.shape(K)[0], tf.float64)

        eye = tf.eye(N, dtype=K.dtype)
        jitter = eye * 1e-6

        Sigma_a = self.alpha * K + (1.0 - self.alpha) * Sigma_f
        Sigma_a_inv = tf.linalg.inv(Sigma_a + jitter)
        if self.loss_type == "MSE":
            q1 = tf.linalg.inv(self.alpha * Sigma_a_inv + eye * self.lambda_i)
            q2 = tf.matmul(eye * self.lambda_i, self.y)
        else:
            q1 = tf.linalg.inv(self.alpha * Sigma_a_inv + eye / self.sigma ** 2 * self.lambda_i)
            q2 = tf.matmul(eye / self.sigma ** 2 * self.lambda_i, self.y)
        mu_f = tf.matmul(q1, q2)
        self.mu_f = mu_f
        self.Sigma_f = Sigma_f
        self.K = K

    def stable_solve(self, mat, rhs):
        jitter = tf.eye(tf.shape(mat)[0], dtype=mat.dtype) * 1e-6
        L = tf.linalg.cholesky(mat + jitter)
        return tf.linalg.cholesky_solve(L, rhs)

    def internal_opt(self, n_it=10):
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        for i in range(n_it):
            with tf.GradientTape(persistent=True) as tape:
                loss = self.train_step()
            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.train_step, self.trainable_variables)

    def predict_out(self, X):
        K_xold_xold = self.kernel(self.X, self.X)
        K_xold_xnew = self.kernel(X, self.X)
        jitter = 1e-6 * tf.eye(tf.shape(K_xold_xold)[0], dtype=K_xold_xold.dtype)
        K_inv = tf.linalg.inv(K_xold_xold + jitter)

        #compute variance
        f_var = self.kernel(X,X) - K_xold_xnew @ K_inv @ tf.transpose(K_xold_xnew)+ K_xold_xnew @ K_inv @ self.Sigma_f @ K_inv @ tf.transpose(K_xold_xnew)
        return K_xold_xnew @ K_inv @ self.mu_f, tf.linalg.diag_part(f_var)

    def compute_lambda_stat(self, n_samples=100):
        # sample_y returns (n_samples, N)
        y_sampl = self.sample_y(n_samples)  # (n_samples, N)
        mu = tf.reshape(self.mu_f, [-1])  # (N,)
        variance = tf.linalg.diag_part(self.Sigma_f)
        if self.loss_type != "MSE":
            variance += self.sigma ** 2
        # per-sample mean squared error (one scalar per sampled function)
        s = tf.reduce_mean((y_sampl - tf.reshape(mu, [1, -1])) ** 2 / variance, axis=1)  # (n_samples,)
        # scalar mean squared residual on training data
        y_vec = tf.reshape(self.y, [-1])  # (N,)
        s_mean = tf.reduce_mean((y_vec - mu) ** 2 / variance)  # scalar

        return tf.reduce_mean(tf.cast(s < s_mean, tf.float32))

    def pp_alg(self,lambda_values = [1., 0.1, 0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.5, 2., 10.]):
        test_values = []
        for lambda_i in lambda_values:
            self.lambda_i = lambda_i
            self.internal_opt()
            self.predict_ins()
            test_values.append(self.compute_lambda_stat())

        ##choose lambda with best
        self.lambda_i = lambda_values[test_values.index(min(test_values))]
        ##fit model with that lambda
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.train_step, self.trainable_variables)



class GVI_SGPR_alpha(gpflow.models.BayesianModel):
    def __init__(self, X, y, inducing, N_u, objective="GVI",loss_type="MSE", structure="Tri", alpha=0.5, test_size=0.1):
        super().__init__()
        self.X = X  # [N, D]
        self.y = y  # [N, 1]
        self.inducing_variable = inducing
        self.N_u = N_u

        self.objective = objective
        # Variational and GP parameters
        self.alpha = gpflow.Parameter(alpha, trainable=False)
        self.loss_type = loss_type
        if structure == "Diag":
            self.Var_q = gpflow.Parameter([1.] * N_u, transform=gpflow.utilities.positive())
        else:
            self.Var_q = gpflow.Parameter(tf.linalg.eye(N_u), transform=gpflow.utilities.triangular())
        self.test_size = test_size
        self.structure = structure
        self.variance = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())
        self.lengthscale = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())
        if loss_type != "MSE":
            self.sigma = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())
        self.lambda_i = 1.

    def stable_solve(self, mat, rhs):
        jitter = tf.eye(tf.shape(mat)[0], dtype=mat.dtype) * 1e-6
        L = tf.linalg.cholesky(mat + jitter)
        return tf.linalg.cholesky_solve(L, rhs)

    def GVI_step(self):
        # Build kernel matrices
        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(self.variance, self.lengthscale)
        K_uu = kernel.matrix(self.inducing_variable, self.inducing_variable)
        K_uf = kernel.matrix(self.inducing_variable, self.X)
        K_fu = tf.transpose(K_uf)

        # Constants
        N = tf.cast(tf.shape(self.X)[0], tf.float64)
        I_M = tf.eye(self.N_u, dtype=tf.float64)

        # Invert via stable_solve
        jitter = tf.eye(self.N_u, dtype=tf.float64) * 1e-6
        K_uu_inv = tf.linalg.inv(K_uu + jitter)

        # Variational covariances
        if self.structure == "Diag":
            Sigma_u = tf.linalg.diag(self.Var_q)
        else:
            Sigma_u = self.Var_q @ tf.transpose(self.Var_q)

        Sigma_a = self.alpha * K_uu + (1.0 - self.alpha) * Sigma_u
        Sigma_a_inv = tf.linalg.inv(Sigma_a + jitter)

        # Compute mu_u
        if self.loss_type == "MSE":
            precision_mid = self.alpha * Sigma_a_inv + (K_uu_inv @ K_uf @ K_fu @ K_uu_inv) * self.lambda_i
            q1 = tf.linalg.inv(precision_mid + jitter)
            q2 = tf.matmul(K_uu_inv @ K_uf, self.y) * self.lambda_i
        else:
            precision_mid = self.alpha * Sigma_a_inv + (
                        K_uu_inv @ K_uf @ K_fu @ K_uu_inv) / self.sigma ** 2 * self.lambda_i
            q1 = tf.linalg.inv(precision_mid + jitter)
            q2 = tf.matmul(K_uu_inv @ K_uf, self.y) / (self.sigma ** 2) * self.lambda_i
        # Compute mu_u
        mu_u = tf.matmul(q1, q2)

        # Log-determinants
        ld_K = tf.linalg.logdet(K_uu + jitter)
        ld_Sig_a = tf.linalg.logdet(Sigma_a + jitter)
        ld_Sig_u = tf.linalg.logdet(Sigma_u + jitter)

        # Loss components
        residual = self.y - K_fu @ K_uu_inv @ mu_u
        T1 = K_uu_inv @ (K_uf @ K_fu)
        T2 = T1 @ (K_uu_inv @ Sigma_u)

        if self.loss_type == "MSE":
            term1 = 0.5 * tf.reduce_sum(tf.square(residual)) * self.lambda_i + 0.5 * self.lambda_i * (
                        tf.reduce_sum(kernel.apply(self.X, self.X)) - tf.linalg.trace(T1) + tf.linalg.trace(T2))
        else:
            term1 = 0.5 * tf.reduce_sum(tf.square(residual)) * (
                        self.sigma ** -2) * self.lambda_i + 0.5 * self.lambda_i * (
                                tf.reduce_sum(kernel.apply(self.X, self.X)) - tf.linalg.trace(T1) + tf.linalg.trace(
                            T2)) * self.lambda_i + N * tf.math.log(self.sigma)

        term2 = 0.5 * self.alpha * tf.reduce_sum(mu_u * (Sigma_a_inv @ mu_u))
        term5 = -1.0 / (2.0 * (self.alpha - 1.0)) * (ld_Sig_a + (self.alpha - 1.0) * ld_Sig_u - self.alpha * ld_K)

        loss = term1 + term2 + term5
        return tf.squeeze(loss)

    def pred_mse(self, y, mu_f, sigma_f):
        loss = 0.5 * tf.transpose(y - mu_f) @ (y - mu_f) + 0.5 * tf.reduce_sum(sigma_f)
        return loss

    def pred_log_prob(self, y, mu_f, sigma_y):
        loss = 0.5 * tf.transpose(y - mu_f) / (sigma_y) @ (y - mu_f)
        loss += 0.5 * tf.reduce_sum(tf.math.log(sigma_y))
        loss += 0.5 * self.sigma ** (-2) * tf.reduce_sum(sigma_y+self.sigma**2)
        return loss

    def predictive_step(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size)

        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(self.variance, self.lengthscale)
        K_uu = kernel.matrix(self.inducing_variable, self.inducing_variable)
        K_uf = kernel.matrix(self.inducing_variable, X_train)
        K_fu = tf.transpose(K_uf)

        eye = tf.eye(K_uu.shape[0], dtype=K_uu.dtype)
        jitter = eye * 1e-6
        K_uu_inv = tf.linalg.inv(K_uu + jitter)

        # Variational covariances
        if self.structure == "Diag":
            Sigma_u = tf.linalg.diag(self.Var_q)
        else:
            Sigma_u = self.Var_q @ tf.transpose(self.Var_q)

        Sigma_a = self.alpha * K_uu + (1.0 - self.alpha) * Sigma_u
        Sigma_a_inv = tf.linalg.inv(Sigma_a + jitter)

        if self.loss_type == "MSE":
            precision_mid = self.alpha * Sigma_a_inv + (K_uu_inv @ K_uf @ K_fu @ K_uu_inv) * self.lambda_i
            q1 = tf.linalg.inv(precision_mid + jitter)
            q2 = tf.matmul(K_uu_inv @ K_uf, y_train) * self.lambda_i
        else:
            precision_mid = self.alpha * Sigma_a_inv + (
                        K_uu_inv @ K_uf @ K_fu @ K_uu_inv) / self.sigma ** 2 * self.lambda_i
            q1 = tf.linalg.inv(precision_mid + jitter)
            q2 = tf.matmul(K_uu_inv @ K_uf, y_train) / (self.sigma ** 2) * self.lambda_i
        # Compute mu_u
        mu_u = tf.matmul(q1, q2)

        K_star = kernel.matrix(X_test, self.inducing_variable)
        K_star_star = kernel.matrix(X_test, X_test)

        mu_p = K_star @ K_uu_inv @ mu_u
        sigma_p = tf.linalg.diag_part(tf.linalg.inv(
            K_star_star - K_star @ K_uu_inv @ tf.transpose(K_star) + K_star @ K_uu_inv @ Sigma_u @ tf.transpose(
                K_star @ K_uu_inv)))

        #
        if self.loss_type == "MSE":
            term1 = self.pred_mse(y_test, mu_p, sigma_p)
        else:
            sigma_p += (self.sigma ** (-2) * tf.eye(tf.cast(tf.shape(K_star_star)[0], tf.float64), dtype=tf.float64))
            sigma_p = tf.linalg.inv(sigma_p)

            term1 = self.pred_log_prob(y_test, mu_p, sigma_p)

        return term1

    def predict_ins(self):
        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(self.variance, self.lengthscale)
        K_uu = kernel.matrix(self.inducing_variable, self.inducing_variable)
        K_uf = kernel.matrix(self.inducing_variable, self.X)
        K_fu = tf.transpose(K_uf)

        # Constants
        N = tf.cast(tf.shape(self.X)[0], tf.float64)
        I_M = tf.eye(self.N_u, dtype=tf.float64)

        # Invert via stable_solve
        jitter = tf.eye(self.N_u, dtype=tf.float64) * 1e-6
        K_uu_inv = tf.linalg.inv(K_uu + jitter)

        # Variational covariances
        if self.structure == "Diag":
            Sigma_u = tf.linalg.diag(self.Var_q)
        else:
            Sigma_u = self.Var_q @ tf.transpose(self.Var_q)
        Sigma_a = self.alpha * K_uu + (1.0 - self.alpha) * Sigma_u
        Sigma_a_inv = tf.linalg.inv(Sigma_a + jitter)

        # Compute mu_u
        if self.loss_type == "MSE":
            precision_mid = self.alpha * Sigma_a_inv + (K_uu_inv @ K_uf @ K_fu @ K_uu_inv) * self.lambda_i
            q1 = tf.linalg.inv(precision_mid + jitter)
            q2 = tf.matmul(K_uu_inv @ K_uf, self.y) * self.lambda_i
        else:
            precision_mid = self.alpha * Sigma_a_inv + (
                        K_uu_inv @ K_uf @ K_fu @ K_uu_inv) / self.sigma ** 2 * self.lambda_i
            q1 = tf.linalg.inv(precision_mid + jitter)
            q2 = tf.matmul(K_uu_inv @ K_uf, self.y) / (self.sigma ** 2) * self.lambda_i

        self.mu_u = tf.matmul(q1, q2)
        self.Sigma_u = Sigma_u

    def predict_out(self, Xnew):
        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(self.variance, self.lengthscale)
        K_uu = kernel.matrix(self.inducing_variable, self.inducing_variable)
        K_xu = kernel.matrix(Xnew, self.inducing_variable)
        I_M = tf.eye(self.N_u, dtype=tf.float64)

        K_uu_inv = self.stable_solve(K_uu, I_M)

        # compute variance
        f_var = kernel.matrix(Xnew, Xnew) - K_xu @ K_uu_inv  @ tf.transpose(
            K_xu) + K_xu @ K_uu_inv @ self.Sigma_u @ K_uu_inv @ tf.transpose(
            K_xu)

        return K_xu @ K_uu_inv @ self.mu_u,tf.linalg.diag_part(f_var)

    @tf.function
    def train_step(self):
        if self.objective == "GVI":
            loss = self.GVI_step()
        else:
            loss = self.predictive_step()
        return loss


    def internal_opt(self):
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        for i in range(10):
            with tf.GradientTape(persistent=True) as tape:
                loss = self.train_step()
            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.train_step, self.trainable_variables)


    def maximum_log_likelihood_objective(self):
        pass

    def sample_y(self, n_samples=100):
        N = self.X.shape[0]
        # compute the density of the sample based on q_u(\mu_u,\Sigma_u) and K

        # put the
        if self.loss_type == "MSE":
            return f
        else:
            noise = tf.random.normal(
                shape=(n_samples, N),
                mean=0.0,
                stddev=self.sigma,  # can be Parameter/Variable
                dtype=tf.float64
            )
            return f + noise

    def compute_lambda_stat(self, n_samples=100):
        y_sampl = self.sample_y(n_samples)
        #
        mu_f = self.predict_out(self.X)
        s = tf.reduce_mean(
            (y_sampl - mu_f) ** 2, axis=1)

        s_mean = tf.reduce_mean((self.y - mu_f) ** 2, axis=0)
        return tf.reduce_mean(tf.cast(s < s_mean, tf.float32))

    def pp_alg(self):
        lambda_values = [1., 0.1, 0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.5, 2., 10.]
        test_values = []
        for lambda_i in lambda_values:
            self.lambda_i = lambda_i
            optimizer = gpflow.optimizers.Scipy()
            optimizer.minimize(self.train_step, self.trainable_variables)
            self.predict_ins()
            test_values.append(self.compute_lambda_stat())

        ##choose lambda with best
        self.lambda_i = lambda_values[test_values.index(min(test_values))]
        ##fit model with that lambda
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.train_step, self.trainable_variables)



####fisher divergence


class GVI_GP_FI_alpha(gpflow.models.BayesianModel):
    def __init__(self, X, y, inducing, alpha=0.5, lengthscale=1., variance=1.):
        super().__init__()
        self.X = X
        self.y = y

        self.alpha = gpflow.Parameter(alpha, trainable=False)
        # GP parameters
        self.kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscale, variance=variance)
        self.inducing_variable = inducing
        self.lambda_i = gpflow.Parameter(10, transform=gpflow.utilities.positive(), trainable=False)
        # self.sigma=tf.Variable(0.3,dtype=tf.float64,trainable=False)
        self.a_p = tf.constant(1.0, dtype=tf.float64)
        self.b_p = tf.constant(1.0, dtype=tf.float64)

    def stable_solve(self, mat, rhs):
        jitter = tf.eye(tf.shape(mat)[0], dtype=mat.dtype) * 1e-6
        L = tf.linalg.cholesky(mat + jitter)
        return tf.linalg.cholesky_solve(L, rhs)

    def maximum_log_likelihood_objective(self):
        pass

    def train_step(self):
        K = self.kernel(self.X, self.X)
        K_uu = self.kernel(self.inducing_variable, self.inducing_variable)
        K_uf = self.kernel(self.inducing_variable, self.X)
        K_fu = tf.transpose(K_uf)

        K_uu_inv = tf.linalg.inv(K_uu + tf.linalg.eye(K_uu.shape[0], dtype=tf.float64) * gpflow.default_jitter())

        Sigma_f = K_fu @ K_uu_inv @ K_uf

        N = tf.cast(tf.shape(K)[0], tf.float64)
        print(N.dtype)
        eye = tf.eye(N, dtype=K.dtype)
        jitter = eye * 1e-6

        Sigma_a = self.alpha * K + (1.0 - self.alpha) * Sigma_f
        Sigma_a_inv = tf.linalg.inv(Sigma_a + jitter)
        q1 = tf.linalg.inv(self.alpha * Sigma_a_inv + eye * (self.lambda_i ** 2))
        q2 = tf.matmul(eye * self.lambda_i ** 2, self.y)
        mu_f = tf.matmul(q1, q2)

        # Log-determinants
        ld_Sig_a = tf.linalg.logdet(Sigma_a + jitter)
        ld_Sig_f = tf.linalg.logdet(Sigma_f + jitter)
        ld_K = tf.linalg.logdet(K + jitter)

        #
        self.lambda_i.assign((N + 1 )/ (tf.reduce_sum((self.y - mu_f) ** 2 + tf.linalg.trace(Sigma_f))))

        # Loss terms
        term1 = tf.transpose(self.y - mu_f) @ (eye * self.lambda_i ** 2) @ (self.y - mu_f)
        term4 = self.lambda_i ** 2 * tf.linalg.trace(Sigma_f) - N * self.lambda_i
        term3 = -tfp.distributions.Gamma(self.a_p, self.b_p).log_prob(self.lambda_i)

        term2 = 0.5 * self.alpha * tf.transpose(mu_f) @ (Sigma_a_inv) @ (mu_f)
        term5 = -1.0 / (2.0 * (self.alpha - 1.0)) * (ld_Sig_a + (self.alpha - 1.0) * ld_Sig_f - self.alpha * ld_K)
        loss = tf.squeeze(term1 + term2 + term3 + term4 + term5)
        self.lambda_i.assign(N + 1 / (tf.reduce_sum((self.y - mu_f) ** 2 + tf.linalg.trace(Sigma_f))))
        return loss

    def predict_ins(self):
        K = self.kernel(self.X, self.X)
        K_uu = self.kernel(self.inducing_variable, self.inducing_variable)
        K_uf = self.kernel(self.inducing_variable, self.X)
        K_fu = tf.transpose(K_uf)

        K_uu_inv = tf.linalg.inv(K_uu + tf.linalg.eye(K_uu.shape[0], dtype=tf.float64) * gpflow.default_jitter())

        Sigma_f = K_fu @ K_uu_inv @ K_uf

        N = tf.cast(tf.shape(K)[0], tf.float64)
        print(N.dtype)
        eye = tf.eye(N, dtype=K.dtype)
        jitter = eye * 1e-6

        Sigma_a = self.alpha * K + (1.0 - self.alpha) * Sigma_f
        Sigma_a_inv = tf.linalg.inv(Sigma_a + jitter)
        q1 = tf.linalg.inv(self.alpha * Sigma_a_inv + eye * (self.lambda_i ** 2))
        q2 = tf.matmul(eye * self.lambda_i ** 2, self.y)
        mu_f = tf.matmul(q1, q2)
        self.mu_f = mu_f
        self.Sigma_f = Sigma_f
        self.K = K



    def internal_opt(self, n_it=10):
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        for i in range(n_it):
            with tf.GradientTape(persistent=True) as tape:
                loss = self.train_step()
            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.train_step, self.trainable_variables)



    def predict_out(self, X):
        K_xold_xold = self.kernel(self.X, self.X)
        K_xold_xnew = self.kernel(X, self.X)
        jitter = 1e-6 * tf.eye(tf.shape(K_xold_xold)[0], dtype=K_xold_xold.dtype)
        K_inv = tf.linalg.inv(K_xold_xold + jitter)
        return K_xold_xnew @ K_inv @ self.mu_f




