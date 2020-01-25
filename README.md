# Gaussian Processes and Bayesian Neural Networks

## Description
This repository contains an example program to demonstrate the functionality of Gaussian processes and
Bayesian Neural Networks who approximate Gaussian Processes under certain conditions, as shown by (Gal, 2016): https://arxiv.org/pdf/1506.02142.pdf.
In this example, a Gaussian Process for a simple regression task is implemented to demonstrate its prior and posterior functio distribution.
Then, a Bayesian Neural Network is trained which approximates the Gaussian process by variational inference.
Again, the posterior distribution is plotted.

## Setup
* You need python 3 and it is recommended to use virtualenv to set up a virtual environment.
    * Create a virtual environment with `virtualenv .venv`
    * Activate it with `activate .venv/bin/activate`
    * In this repository run `pip install -r requirements` to install the pip packages
* Run the program with `python regression_with_GP.py`and `python regression_with_BNN.py`

## Contents
*`regression_with_GP.py`: Perform regression with a Gaussian process. Plot the prior and posterior distribution.
*`regression_with_BNN.py`: Perform regression with a GP and a BNN. Plots the posterior distributions in the folder ./output/ .
