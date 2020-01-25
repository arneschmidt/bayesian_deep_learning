import tensorflow as tf
import numpy as np
from regression_plot import training_plot


class BayesianNeuralNetwork:
    """
    Class containing the implementation of a Bayesian neural network with one hidden layer for regression.
    """
    def __init__(self, neurons_hidden_layer: int, initializer: str, sigma_observation_noise: float,
                 sigma_w1: float, sigma_b1: float,
                 sigma_w2: float, sigma_b2: float,
                 sigma_w3: float, sigma_b3: float,
                 sigma_w4: float):
        """
        Initialize network with given hyperparameters.
        :param number_of_neurons: number of neurons in the hidden layer
        :param sigma_observation_noise: standard deviation of the observation noise
        :param sigma_1: standard deviation of the Gaussian distribution of weights W_1
        :param sigma_b: standard deviation of the Gaussian distribution of weights b
        :param sigma_2: standard deviation of the Gaussian distribution of weights W_2
        """
        self.session = tf.Session()
        self.neurons_hidden_layer = neurons_hidden_layer
        self.initializer = initializer
        self.sigma_observation_noise = sigma_observation_noise
        self.sigma_w1 = sigma_w1
        self.sigma_b1 = sigma_b1
        self.sigma_w2 = sigma_w2
        self.sigma_b2 = sigma_b2
        self.sigma_w3 = sigma_w3
        self.sigma_b3 = sigma_b3
        self.sigma_w4 = sigma_w4
        self.variable_scope = tf.get_variable_scope()

    def tf_graph(self, x_points: tf.Tensor):
        """
        Tensorflow graph of the Bayesian neural network.
        :param x_points: x values of training points
        :return: y_predicted (output) and mean of weight distributions for regularization
        """
        x_points=tf.reshape(x_points, [-1,1])

        weight_initializer = tf.truncated_normal_initializer()

        # hidden layer 1
        mu_w1 = tf.get_variable(name='mu_w1', shape=[self.neurons_hidden_layer], initializer=weight_initializer)
        w1_sample = mu_w1 + tf.random_normal(shape=mu_w1.get_shape(), mean=0, stddev=self.sigma_w1, dtype=tf.float32)

        mu_b1 = tf.get_variable(name='mu_b1', shape=[self.neurons_hidden_layer], initializer=weight_initializer)
        b1_sample = mu_b1 + tf.random_normal(shape=mu_b1.get_shape(), mean=0, stddev=self.sigma_b1, dtype=tf.float32)
        phi1 = tf.sqrt(1/self.neurons_hidden_layer) * tf.nn.relu(tf.multiply(x_points, w1_sample) + b1_sample)

        # hidden layer 2
        mu_w2 = tf.get_variable(name='mu_w2', shape=[self.neurons_hidden_layer, self.neurons_hidden_layer], initializer=weight_initializer)
        w2_sample = mu_w2 + tf.random_normal(shape=mu_w2.get_shape(), mean=0, stddev=self.sigma_w2, dtype=tf.float32)

        mu_b2 = tf.get_variable(name='mu_b2', shape=[self.neurons_hidden_layer], initializer=weight_initializer)
        b2_sample = mu_b2 + tf.random_normal(shape=mu_b2.get_shape(), mean=0, stddev=self.sigma_b2, dtype=tf.float32)
        phi2 = tf.sqrt(1/self.neurons_hidden_layer) * tf.nn.relu(tf.matmul(phi1, w2_sample) + b2_sample)

        # hidden layer 3
        mu_w3 = tf.get_variable(name='mu_w3', shape=[self.neurons_hidden_layer, self.neurons_hidden_layer], initializer=weight_initializer)
        w3_sample = mu_w3 + tf.random_normal(shape=mu_w3.get_shape(), mean=0, stddev=self.sigma_w3, dtype=tf.float32)

        mu_b3 = tf.get_variable(name='mu_b3', shape=[self.neurons_hidden_layer], initializer=weight_initializer)
        b3_sample = mu_b3 + tf.random_normal(shape=mu_b3.get_shape(), mean=0, stddev=self.sigma_b3, dtype=tf.float32)
        phi3 = tf.sqrt(1/self.neurons_hidden_layer) * tf.nn.relu(tf.matmul(phi2, w3_sample) + b3_sample)

        # output layer
        mu_w4 = tf.get_variable(name='mu_w4', shape=[self.neurons_hidden_layer, 1], initializer=weight_initializer)
        w4_sample = mu_w4 + tf.random_normal(shape=mu_w4.get_shape(), mean=0, stddev=self.sigma_w4, dtype=tf.float32)

        y_predicted = tf.reshape(tf.matmul(phi3, w4_sample), [-1])
        return y_predicted, mu_w1, mu_b1, mu_w2, mu_b2, mu_w3, mu_b3, mu_w4

    def train(self, x_train: np.array, y_train: np.array, train_steps: int):
        """
        Train the bayesian network for a given number of training steps.
        :param x_train: training points, x values
        :param y_train: training points, y values
        :param train_steps: number of training steps
        """
        with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE):
            # placeholder for training data
            x = tf.placeholder(tf.float32, [None])
            y = tf.placeholder(tf.float32, [None])

            # network prediction
            y_predicted, mu_w1, mu_b1, mu_w2, mu_b2, mu_w3, mu_b3, mu_w4 = self.tf_graph(x)

            # calculate loss function
            mean_squared_error = tf.nn.l2_loss(y - y_predicted)
            regularization = tf.nn.l2_loss(mu_w1) + tf.nn.l2_loss(mu_b1) + tf.nn.l2_loss(mu_w2) + tf.nn.l2_loss(mu_b2) \
                             + tf.nn.l2_loss(mu_w3) + tf.nn.l2_loss(mu_b3) + tf.nn.l2_loss(mu_w4)

            loss = (mean_squared_error + (self.sigma_observation_noise**2) * regularization)/x_train.size

            # define training step with stochastic gradient descent
            train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(loss)

            # initialize variables and train
            self.session.run(tf.global_variables_initializer())
            for i in  range(train_steps):
                self.session.run(train_step, feed_dict={x: x_train, y: y_train})
                if i%100 == 0:
                    print("Iteration:", i, "Loss: ", self.session.run([loss], feed_dict={x: x_train, y: y_train}))
                if i%10000 == 0:
                    x_function_plot = np.arange(-25, 25, 0.1)
                    y_mean, y_std = self.predict(x_function_plot, 100)
                    training_plot("BNN_training_plot", i, x_function_plot, x_train, y_train, y_mean, y_std )

    def predict(self, x_test: np.array, number_of_samples: int) -> np.array:
        """
        Make a prediction with the trained network for a given test input.
        :param x_test: x values of the test input
        :param number_of_samples: number of samples for Monte Carlo integration
        :return: predictive distribution (mean, std.) of the network
        """
        with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE):
            x = tf.placeholder(tf.float32, [None])
            # range of the function over the X axis
            y_samples = [self.tf_graph(x)[0]]
            for i in range(number_of_samples - 1):
                y_samples = tf.concat([y_samples, [self.tf_graph(x)[0]]], 0)

            mean, var = tf.nn.moments(y_samples, axes=[0])
            std = tf.sqrt(var)
            mean_out, std_out = self.session.run([mean, std], feed_dict={x: x_test})
            std_out = std_out + self.sigma_observation_noise**2

        return mean_out, std_out
