
"""
Data Simulation for Anfis Sandbox
"""
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from .markov_process import MRS
from .star_process import STAR
import matplotlib.pyplot as plt
plt.rcParams['axes.xmargin'] = 0  # remove margins from all plots
##############################################################################


def get_data_name(data_id):
    data_sets = ['mackey', 'sinc',
                 'threeInputNonlin', 'MarkovRS', 'TAR', 'STAR']

    return data_sets[data_id]


def split_data(X, y, batch_size):
    # adjust X and y for batch_size
    adj_id = np.arange(len(y) - len(y) % batch_size)
    X, y = X[adj_id, :], y[adj_id]

    # split test & train according to batches
    batches = len(y) / batch_size
    train_batches = np.round(batches * 0.6) / batches

    train_id = np.arange(len(y) * train_batches, dtype=int)
    test_id = np.arange(len(y) * train_batches, len(y), dtype=int)

    X_train, y_train, X_test, y_test = X[train_id,
                                         :], y[train_id], X[test_id, :], y[test_id]

    return X, X_train, X_test, y, y_train, y_test


def gen_data(data_id, n_obs, n_input, batch_size=16, lag=1):

    # Mackey
    if data_id == 0:
        y = mackey(124 + n_obs + n_input)[124:]
        X, y = gen_X_from_y(y, n_input, lag)

    # Nonlin sinc equation
    elif data_id == 1:
        X, y = sinc_data(n_obs)
        assert n_input == 2, 'Nonlin sinc equation data set requires n_input==2. Please chhange to 2.'

    # Nonlin three-input equation
    elif data_id == 2:
        X, y = nonlin_data(n_obs)
        assert n_input == 3, 'Nonlin Three-Input Equation required n_input==3. Please switch to 3.'

    # Markov Regime switching ts
    elif data_id == 3:
        mrs_model = MRS(P=np.array([[0.985, 0.01, 0.004],
                                    [0.03, 0.969, 0.001],
                                    [0.00, 0.03, 0.97]]))
        mrs_model.sim(n_obs + n_input)
        mrs_model.plot(colored=True)
        X, y = gen_X_from_y(mrs_model.r, n_input, lag)

    # Threshold Autoregressive Model (TAR)
    elif data_id == 4:
        mu_params = np.array([0.05, 0.05])
        sigma_params = np.array([0.20, 0.20])
        AR_params = np.array([[0.55, 0],
                              [-0.30, 0]])
        gamma = float('inf')
        star_model = STAR(mu_params, sigma_params, AR_params, gamma)
        star_model.sim(n_obs + n_input)
        star_model.plot(colored=True)
        X, y = gen_X_from_y(star_model.r, n_input, lag)

    # Smooth Transition Autoregressive Model (STAR)
    elif data_id == 5:
        mu_params = np.array([0.15, -0.10])
        sigma_params = np.array([0.0, 0.0])
        AR_params = np.array([[0.55, 0],
                              [-0.30, 0]])
        gamma = 1
        star_model = STAR(mu_params, sigma_params, AR_params, gamma)
        star_model.sim(n_obs + n_input)
        star_model.plot(colored=True)
        X, y = gen_X_from_y(star_model.r, n_input, lag)

    # standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaler = StandardScaler()
    y = scaler.fit_transform(y)

    # split data into test and train set
    X, X_train, X_test, y, y_train, y_test = split_data(X, y, batch_size)

    return X, X_train, X_test, y, y_train, y_test

##############################################################################
# Mackey-Glass series computation


def mackey(n_iters):
    x = np.zeros((n_iters,))
    x[0:30] = 0.23 * np.ones((30,))
    t_s = 30
    for i in range(30, n_iters - 1):
        a = x[i]
        b = x[i - t_s]
        y = ((0.2 * b) / (1 + b ** 10)) + 0.9 * a
        x[i + 1] = y
    return x

# Modelling a two-Input Nonlinear Function (Sinc Equation)


def sinc_equation(x1, x2):
    return ((np.sin(x1) / x1) * (np.sin(x2) / x2))


def sinc_data(n_obs, multiplier=2, noise=False):
    X = (np.random.rand(n_obs, 2) - .5) * multiplier
    y = sinc_equation(X[:, 0], X[:, 1]).reshape(-1, 1)
    if noise == True:
        y = y + np.random.randn(n_obs) * 0.1
    return X.astype('float32'), y.astype('float32')


# Modelling a Three-Input Nonlinear Function (Sinc Equation)
def nonlin_equation(x, y, z):
    return ((1 + x**0.5 + 1 / y + z**(-1.5))**2)


def nonlin_data(n_obs, multiplier=1, noise=False):
    X = np.random.rand(n_obs, 3) * multiplier + 1
    y = nonlin_equation(X[:, 0], X[:, 1], X[:, 2]).reshape(-1, 1)
    if noise == True:
        y = y + np.random.randn(n_obs)
    return X.astype('float32'), y.astype('float32')


# Generate a input matrix X from time series y
def gen_X_from_y(x, n_input=1, lag=1):
    n_obs = len(x) - n_input * lag

    data = np.zeros((n_obs, n_input + 1))
    for t in range(n_input * lag, n_obs + n_input * lag):
        data[t - n_input * lag, :] = [x[t - i * lag]
                                      for i in range(n_input + 1)]
    X = data[:, 1:].reshape(n_obs, -1)
    y = data[:, 0].reshape(n_obs, 1)

    return X.astype('float32'), y.astype('float32')
