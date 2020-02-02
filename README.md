# Gaussian Processes and Bayesian Neural Networks

## Description
This repository contains an example program to demonstrate the functionality of Gaussian processes and
Bayesian Neural Networks who approximate Gaussian Processes under certain conditions, as shown by (Gal, 2016): https://arxiv.org/pdf/1506.02142.pdf. \
In this example, a Gaussian Process for a simple regression task is implemented to demonstrate its prior and posterior function distribution.
Then, a Bayesian Neural Network is trained which approximates the Gaussian process by variational inference.
Again, the posterior distribution is plotted.


### Problem Setup
We assume that we want to approximate a real-world function f(x) which describes a relationship between data coordinates x and observations y.
The function is corrupted with random Gaussian observation noise.

### Gaussian Processes (`regression_with_GP.py`)
This script shows, how regression with Gaussian processes looks like for this problem. 
The Gaussian process is defined by a kernel function, in this example a squared exponential kernel (function k_se) which is a common choice.  
Before observing the data, the Gaussian process has a prior probability over functions.  
  
![Gaussian process prior probability](./output/GP_prior.png) 
**Figure 1: Gaussian prior probability distribution over functions and three samples drawn from this probability distribution.**  
  
After observing the data, the posterior distribution can be calculated which can be derived with Bayes' formula.
The posterior function combines the prior belief with the observed data.
  
![Gaussian process posterior probability](./output/GP_posterior.png) 
**Figure 2: Gaussian posterior probability distribution over functions and two test points with variance**  
  
Notice that the variance of the distribution is higher in regions where no data points are present. For two example test points the mean and variance prediction is shown.  
The variance gives information about the model uncertainty which can be very valuable in safety critical environments.

### Bayesian Neural Networks (`regression_with_BNN.py`)
Now we want to see how Bayesian neural networks can approximate Gaussian processes. The kernel of the Gaussian process depends on the activation function of the neural network.  
First, a normal GP with that kernel function is defined. As in the previous example, the GP posterior is calculated and plotted:

![Gaussian process posterior probability](./output/BNN_full_GP_posterior.png) 
**Figure 3: Gaussian posterior probability distribution over functions and two test points with variance**  
  
Now we define a Bayesian neural network with one hidden layers. It has a Gaussian normal probability distribution over its weights and biases.
By variational inference we approximate the Gaussian process posterior probability during training.
The posterior distribution of the BNN is again plotted:

![BNN posterior probability](./output/BNN_variational_inference_GP_posterior.png) 
**Figure 4: BNN posterior probability distribution over functions and two test points with variance.**  
  
We see that the Bayesian neural networks incorporates the ability to show the model uncertainty as it does not define a deterministic function but a probability distribution over functions!
Additionally we showed how they approximate Gaussian processes, which are mathematically well understood.

## Setup
* You need python 3 and it is recommended to use virtualenv to set up a virtual environment.
    * Create a virtual environment with `virtualenv .venv`
    * Activate it with `activate .venv/bin/activate`
    * In this repository run `pip install -r requirements` to install the pip packages
* Run the program with `python regression_with_GP.py`and `python regression_with_BNN.py`


*: Perform regression with a Gaussian process. Plot the prior and posterior distribution.
*`regression_with_BNN.py`: Perform regression with a GP and a BNN. Plots the posterior distributions in the folder ./output/ .
