import numpy as np
from gaussian_process import GaussianProcess
from bayesian_network import BayesianNeuralNetwork
from regression_plot import posterior_plot

def f(x):
    """ Objective function f(x) for data generation"""
    return 1/(1+np.exp(-x)) + 0.001*x**2

def k(x_1: np.array, x_2: np.array):
    """ Covariance matrix of Gaussian process defined by Bayesian neural network"""
    np.random.seed(1)
    number_of_mc_samples = 100
    w_samples_1 = np.random.normal(loc=0.0, scale=1, size=(number_of_mc_samples))
    b_samples_1 = np.random.normal(loc=0.0, scale=1, size=(number_of_mc_samples))
    Phi_1 = np.maximum(0, np.multiply.outer(x_1.T, w_samples_1) + b_samples_1)
    Phi_2 = np.maximum(0, np.multiply.outer(x_2.T, w_samples_1) + b_samples_1)
    K = (1/number_of_mc_samples * np.matmul(Phi_1, Phi_2.T))
    return K


def main():
    # Define training points
    x_train = np.array([-15,  -4,  4, 9, 13, 18])
    np.random.seed(1)
    noise = np.random.normal(scale=0.1, size=x_train.size)
    y_train = f(x_train) + noise # function with observation noise

    x_test = np.array([-20, 10])
    x_function_plot = np.arange(-25, 25, 0.1)

    # Full Gaussian Process
    GP = GaussianProcess(x_train=x_train, y_train=y_train, covariance_matrix_function=k, sigma=0.1)

    # get mean and variance function of Gaussian process
    y_gp_mean_function, y_gp_std_function = GP.predictive_distribution(x_function_plot)
    # make predictions for two test points
    y_gp_test_mean, y_gp_test_std = GP.predictive_distribution(x_test)

    posterior_plot("BNN_full_GP_posterior", x_function_plot, f(x_function_plot), x_train, y_train,
                   y_gp_mean_function, y_gp_std_function,
                   x_test, y_gp_test_mean, y_gp_test_std)

    # Variational Inference Approximation of Gaussian Process
    bnn = BayesianNeuralNetwork(neurons_hidden_layer=100, initializer='normal', sigma_observation_noise=0.1,
                                sigma_w1=0.01, sigma_b1=0.01,
                                sigma_w2=0.02)
    bnn.train(x_train, y_train, train_steps=20000)

    # calculate mean and variance function
    y_vi_mean_function, y_vi_std_function = bnn.predict(x_function_plot, 100)
    # make predictions for two test points
    y_vi_test_mean, y_vi_test_std = bnn.predict(x_test, 100)

    # plot results
    posterior_plot("BNN_variational_inference_GP_posterior", x_function_plot, f(x_function_plot), x_train, y_train,
                   y_vi_mean_function, y_vi_std_function,
                   x_test, y_vi_test_mean, y_vi_test_std)

if __name__ == "__main__": main()