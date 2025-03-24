import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import random
import w  # Assuming `w` is a module providing IMQ functionality


class GaussianProcessModel:
    def __init__(self, variance=4.5, lengthscale=0.9, alpha=0.5, sigma=0.01, c_f=1.0,prop=0.1):
        self.variance = tf.Variable(variance, dtype=tf.float64, constraint=lambda z: tf.clip_by_value(z, 0.001, 10))
        self.lengthscale = tf.Variable(lengthscale, dtype=tf.float64,
                                       constraint=lambda z: tf.clip_by_value(z, 0.001, 10))
        self.alpha = tf.Variable(alpha, dtype=tf.float64, constraint=lambda z: tf.clip_by_value(z, -1, 2))
        self.sigma = tf.Variable(sigma, dtype=tf.float64, constraint=lambda z: tf.clip_by_value(z, 0.001, 10))
        self.c_f = tf.Variable(c_f, dtype=tf.float64, constraint=lambda z: tf.clip_by_value(z, 0.001, 10))
        self.kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(self.variance, self.lengthscale)


    def update_mu_f(self, alpha, Sigma_a_inv, J_w_inv, sigma_2, Y_train, mu_w):
        q1 = tf.linalg.inv(alpha * Sigma_a_inv + 2 * J_w_inv / sigma_2)
        q2 = tf.matmul((2 * J_w_inv / sigma_2), (Y_train - mu_w))
        return tf.matmul(q1, q2)

    def update_Sigma_f(self, c_f, K, sigma_2, J_w):
        return c_f * K @ tf.linalg.inv(K + sigma_2 * J_w / 2) * sigma_2 * J_w / 2

    def divide_batch(self, X, y, test_ratio=0.2, seed=None):
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

    def maximize_c(self, y, quant):
        return np.quantile(abs(y), 1 - quant)

    def full_sample_estimate(self, X, Y):
        K = self.kernel.matrix(X, X)
        self.w_full = self.sigma / (2 ** 0.5) * self.w_full
        J_w = tf.linalg.tensor_diag(tf.squeeze(tf.math.pow(self.w_full, -2))) * self.sigma ** 2 / 2
        J_w_inv = tf.linalg.tensor_diag(tf.squeeze(tf.math.pow(self.w_full, 2))) * 2 / (self.sigma ** 2)
        Sigma_f = (self.c_f * K @ tf.linalg.inv(K + self.sigma ** 2 / 2 * J_w) * self.sigma ** 2 / 2) @ J_w
        Sigma_a = self.alpha * K + (1 - self.alpha) * Sigma_f
        Sigma_a_inv = tf.linalg.inv(Sigma_a)
        mu_f = self.update_mu_f(self.alpha, Sigma_a_inv, J_w_inv, self.sigma ** 2, Y, tf.zeros_like(Y))
        return mu_f, Sigma_f

    def train(self, X, y, iterations=500,outl_prop=0.1):
        optimizer = tf.optimizers.Adam()

        def maximize_c(y, quant):
            return np.quantile(abs(y), 1 - quant)
        c = maximize_c(y, outl_prop)
        W = w.IMQ(c)
        self.w_full = tf.cast(W.W(X, y), dtype=tf.float64)
        self.m_w_full = W.dy(X, y) ** 2
        for i in range(iterations):
            with tf.GradientTape(persistent=True) as tape:
                X_train, Y_train, X_test, Y_test, train_indices, test_indices = self.divide_batch(X, y)
                K = self.kernel.matrix(X_train, X_train)
                w_train = tf.gather(self.w_full, train_indices)
                m_w = tf.gather(self.m_w_full, train_indices)
                J_w = tf.linalg.tensor_diag(tf.squeeze(tf.math.pow(w_train, -2)))
                J_w_inv = tf.linalg.tensor_diag(tf.squeeze(tf.math.pow(w_train, 2)))
                Sigma_f = (self.c_f * K @ tf.linalg.inv(K + self.sigma ** 2 * J_w) * self.sigma ** 2 / 2) @ J_w
                Sigma_a = self.alpha * K + (1 - self.alpha) * Sigma_f
                Sigma_a_inv = tf.linalg.inv(Sigma_a)
                mu_f = self.update_mu_f(self.alpha, Sigma_a_inv, J_w_inv, self.sigma ** 2, Y_train, m_w)
                loss = -tf.reduce_sum(tfp.distributions.Normal(mu_f, tf.linalg.diag_part(Sigma_f)).log_prob(Y_test))
            grads = tape.gradient(loss, [self.c_f, self.sigma, self.variance, self.lengthscale, self.alpha])
            optimizer.apply_gradients(zip(grads, [self.c_f, self.sigma, self.variance, self.lengthscale, self.alpha]))
        return self.full_sample_estimate(X, y)
