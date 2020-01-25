import numpy as np


class GaussianProcess:
    """
    Class containing the implementation of a Gaussian process for given trianing data.
    """
    def __init__(self, x_train: np.array, y_train: np.array, covariance_matrix_function, sigma: float):
        """
        Define a Gaussian process on a given training set
        :param x_train: x values of training set
        :param y_train: corresponding y values of training set
        :param covariance_matrix_function: function that returns covariance matrix for two input arrays
        :param sigma: std. of observation noise
        """
        self.x_train = x_train
        self.y_train = y_train
        self.k = covariance_matrix_function
        self.sigma = sigma

    def prior_distribution(self, x: np.array):
        """ Get prior distribution (mean, std) for input points x"""
        k = self.k(x, x)
        prior_std = np.sqrt(np.diag(k))
        prior_mean = np.zeros_like(prior_std)
        return prior_mean, prior_std

    def prior_samples(self, x: np.array, number_of_samples: int):
        """ Get output values of sampled functions from prior"""
        k = self.k(x, x)

        # Compute square root of covariance matrix by cholesky decomposition
        L = np.linalg.cholesky(k + 1e-10 * np.identity(k.shape[0]))
        # Sample Gaussian process realizations by multiplying with random normal
        y_prior_samples = np.dot(L, np.random.normal(size=(k.shape[0], number_of_samples)))
        return y_prior_samples

    def predictive_distribution(self, x_test: np.array):
        """ Prediction of test points, given training data"""

        # Calculate covariance matrices
        k_0 = self.k(self.x_train, self.x_train)
        k_1 = self.k(x_test, self.x_train)
        k_2 = self.k(x_test, x_test)
        k_inverse = np.linalg.inv(k_0 + np.identity(k_0.shape[0])* self.sigma**2)

        # formulas for GP prediction
        mean = k_1.dot(k_inverse).dot(self.y_train)
        covariance_matrix = k_2 + np.identity(k_2.shape[0]) * self.sigma**2 - k_1.dot(k_inverse).dot(np.transpose(k_1))
        standard_deviation = np.sqrt(np.diag(covariance_matrix))
        return mean, standard_deviation
