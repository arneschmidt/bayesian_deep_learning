import matplotlib.pyplot as plt
import numpy as np
import os


def prior_plot(plot_title: str, x_function_plot: np.array, y_prior_samples: np.array,
                       y_prior_mean: np.array, y_prior_std: np.array):
    """
    Plot prior distribution of Gaussian Process
    :param plot_title: title of plot
    :param x_function_plot: x values of function plot
    :param y_prior_samples: function samples of prior, has to be same length as x_function_plot
    :param y_prior_mean: mean function of prior distribution, has to be same length as x_function_plot
    :param y_prior_std: std. function of prior distribution, has to be same length as x_function_plot
    :return:
    """
    if not os.path.exists("output"):
        os.makedirs("output")
    output_path = "output/" + plot_title + ".png"

    plt.figure()
    #plt.title('Prior Distribution')
    plt.ylim([-3, 3])
    plt.xlim([-25, 25])
    plt.xlabel('x-Input')
    plt.ylabel('y-Target Value')
    plt.plot(x_function_plot, y_prior_samples[:,0], 'k-', label='Prior Samples')
    plt.plot(x_function_plot, y_prior_samples[:,1:3], 'k-')
    plt.plot(x_function_plot, y_prior_mean, 'k--', label='Prior mean')
    plt.plot(x_function_plot, y_prior_std, 'k:', label='Prior std.')
    plt.plot(x_function_plot, -y_prior_std, 'k:')
    plt.legend(loc='lower right')
    plt.savefig(output_path)
    print("Prior plot saved: " + output_path)


def posterior_plot(plot_title: str, x_function_plot: np.array, y_generating_function: np.array,
                   x_train: np.array, y_train: np.array,
                   y_mean_function: np.array, y_std_function: np.array,
                   x_test:np.array, y_test_mean: np.array, y_test_std):
    """
    Plot posterior distribution of Gaussian process
    :param plot_title: title of plot
    :param x_function_plot: x values of function plot
    :param y_generating_function: y values of generating function, has to be same length as x_function_plot
    :param x_train: x train datapoints
    :param y_train: y train datapoints
    :param y_mean_function: y values posterior mean function, has to be same length as x_function_plot
    :param y_std_function: y values posterior std function, has to be same length as x_function_plot
    :param x_test: x values of test points
    :param y_test_mean:  y values of posterior mean test points, has to be same length as x_test
    :param y_test_std: y values of posterior std. test points, has to be same length as x_test
    :return:
    """

    if not os.path.exists("output"):
        os.makedirs("output")
    output_path = "output/" + plot_title + ".png"

    plt.figure()
    #plt.title('Predictive distribution and two test predictions')
    plt.xlabel('x-Input')
    plt.ylabel('y-Target Value')
    plt.ylim([-1, 2])
    plt.xlim([-25, 25])
    plt.plot(x_function_plot, y_generating_function, 'k-', label='f(x)')
    plt.plot(x_function_plot, y_mean_function, 'k--', label='mean function')
    plt.plot(x_function_plot, y_mean_function + y_std_function, 'k:', label='std.')
    plt.plot(x_function_plot, y_mean_function - y_std_function, 'k:')
    plt.plot(x_train, y_train, 'kx', label='training points')
    plt.errorbar(x_test, y_test_mean, yerr=y_test_std, fmt='ko', capsize=10, label='test predictions')
    plt.legend(loc='lower right')
    plt.savefig(output_path)
    print("Posterior plot saved: " + output_path)

def training_plot(plot_title: str, training_step: int,  x_function_plot: np.array,
                   x_train: np.array, y_train: np.array,
                   y_mean_function: np.array, y_std_function: np.array):
    """
    Plot posterior distribution of Gaussian process
    :param plot_title: title of plot
    :param training_step: training iteration
    :param x_function_plot: x values of function plot
    :param x_train: x train datapoints
    :param y_train: y train datapoints
    :param y_mean_function: y values posterior mean function, has to be same length as x_function_plot
    :param y_std_function: y values posterior std function, has to be same length as x_function_plot
    :return:
    """

    if not os.path.exists("output"):
        os.makedirs("output")
    output_path = "output/" + plot_title + ".png"

    plt.figure()
    plt.title('Training Step: '+ str(training_step))
    plt.xlabel('x-Input')
    plt.ylabel('y-Target Value')
    plt.ylim([-1, 2])
    plt.xlim([-25, 25])
    plt.plot(x_function_plot, y_mean_function, 'k--', label='mean function')
    plt.plot(x_function_plot, y_mean_function + y_std_function, 'k:', label='std.')
    plt.plot(x_function_plot, y_mean_function - y_std_function, 'k:')
    plt.plot(x_train, y_train, 'kx', label='training points')
    plt.legend(loc='lower right')
    plt.savefig(output_path)
    print("Training plot saved: " + output_path)