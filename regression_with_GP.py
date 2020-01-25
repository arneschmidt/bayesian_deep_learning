import numpy as np
from gaussian_process import GaussianProcess
from regression_plot import prior_plot, posterior_plot

def f(x):
    """ Objective function f(x) for data generation"""
    return 1/(1+np.exp(-x)) + 0.001*x**2

def k_se(x_1: np.array, x_2: np.array):
    """ Covariance matrix with squared exponential covariance function"""
    length_scale = 100
    return np.exp(- np.subtract.outer(x_1, x_2) ** 2 / (2 *length_scale))

def main():
    # define training points with noise (random but fixed for reproduction)
    x_train = np.array([-15,  -4,  4, 9, 13, 18])
    np.random.seed(1)
    noise = np.random.normal(scale=0.1, size=x_train.size)
    y_train = f(x_train) + noise

    # define a dense sample of points to show continous functions
    x_function_plot = np.arange(-25, 25, 0.01)

    # initialize the Gaussian process
    GP = GaussianProcess(x_train=x_train, y_train=y_train, covariance_matrix_function=k_se, sigma=0.1)

    # get prior distribution and three samples
    y_prior_mean, y_prior_std = GP.prior_distribution(x=x_function_plot)
    y_prior_samples = GP.prior_samples(x=x_function_plot, number_of_samples=3)

    # plot results
    prior_plot("GP_prior", x_function_plot, y_prior_samples, y_prior_mean, y_prior_std)

    # get mean and variance function of Gaussian process
    y_mean_function, y_std_function = GP.predictive_distribution(x_function_plot)

    # make predictions for two test points
    x_test = np.array([-20, 10])
    y_test_mean, y_test_std = GP.predictive_distribution(x_test)

    # plot results
    posterior_plot("GP_posterior", x_function_plot, f(x_function_plot), x_train, y_train,
                   y_mean_function, y_std_function,
                   x_test, y_test_mean, y_test_std)

if __name__ == "__main__": main()