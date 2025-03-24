import tensorflow as tf


class ARDKernel(tf.keras.layers.Layer):
    def __init__(self, num_dims, initial_lengthscales=None, initial_variance=1.0, **kwargs):
        """
        Initializes the ARD kernel.

        Parameters:
        - num_dims: The number of input dimensions.
        - initial_lengthscales: Optional. Initial values for the lengthscales.
                                If None, initialized to 1.0 for all dimensions.
        - initial_variance: Initial value for the signal variance (default is 1.0).
        """
        super().__init__(**kwargs)
        self.num_dims = num_dims

        # Trainable lengthscales (one for each dimension)
        if initial_lengthscales is None:
            initial_lengthscales = [1.]*self.num_dims
        self.lengthscales = tf.Variable(initial_lengthscales, trainable=True, name="lengthscales")

        # Trainable variance
        self.variance = tf.Variable(initial_variance, trainable=True, dtype=tf.float32, name="variance")

    def _scaled_square_dist(self, X1, X2):
        """
        Computes the scaled squared distances between points.

        Parameters:
        - X1: Tensor of shape (N, D) (N samples, D dimensions).
        - X2: Tensor of shape (M, D) (M samples, D dimensions).

        Returns:
        - A tensor of shape (N, M) with scaled squared distances.
        """
        X1_scaled = X1 / self.lengthscales
        X2_scaled = X2 / self.lengthscales
        sq_dist = tf.reduce_sum(tf.square(X1_scaled[:, tf.newaxis, :] - X2_scaled[tf.newaxis, :, :]), axis=-1)
        return sq_dist

    def call(self, X1, X2=None):
        """
        Computes the ARD kernel matrix.

        Parameters:
        - X1: Tensor of shape (N, D) of input points.
        - X2: Tensor of shape (M, D) of input points (optional). If None, X2 = X1.

        Returns:
        - A (N, M) kernel matrix.
        """
        if X2 is None:
            X2 = X1
        sq_dist = self._scaled_square_dist(X1, X2)
        return self.variance * tf.exp(-0.5 * sq_dist)

    def diag(self, X):
        """
        Computes the diagonal of the kernel matrix (variance for each point).

        Parameters:
        - X: Tensor of shape (N, D) of input points.

        Returns:
        - A tensor of shape (N,) containing the diagonal elements.
        """
        return tf.fill([tf.shape(X)[0]], self.variance)
