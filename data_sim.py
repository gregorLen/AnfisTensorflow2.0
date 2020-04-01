
"""
Data Simulation for Anfis Training
"""
import numpy as np
from sklearn.datasets import load_diabetes 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

##############################################################################

def gen_data(data_set, n_obs, n_input, n_memb):
    
    scaler = StandardScaler()    # data will be standadized
    
    if data_set == 1: 
        X = np.random.randn(n_obs, n_input).astype('float32')
        X = scaler.fit_transform(X)
        y = np.random.randn(n_obs, 1).astype('float32')*10
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4)    
        
    elif data_set == 2: 
        X, y = mackey_data(n_obs, n_input, 1)
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4)    
        
    elif data_set == 3: 
        X, y = sinc_data(n_obs)
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4)
        if n_input != 2:
            n_input = 2 
            print('Nonlin sinc equation data set requires n_input==3. Switched to 3.')
        
    elif data_set == 4: 
        X, y = nonlin_data(n_obs)
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4)
        if n_input != 3:
            n_input = 3 
            print('Nonlin Three-Input data set requires n_input==3. Switched to 3.')
        
    elif data_set == 5: 
        n_obs = 432
        print('Dataset diabetes is limited to 432 observations')
        X, y = load_diabetes(return_X_y=True)
        X = scaler.fit_transform(X)
        X, y = X[:n_obs,0:n_input].astype('float32'), y[:n_obs].astype('float32').reshape(-1,1)
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.37)

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

def mackey_data(n_obs, n_input=1, D=1, noise=False):
    x = mackey(300+n_obs+n_input*D)[300:]
    data = np.zeros((n_obs, n_input+1))
    for t in range(n_input*D, n_obs + n_input*D):
        data[t - n_input*D,:] = [x[t-i*D] for i in range(n_input+1)]
        X = data[:,1:]
        y = data[:,0].reshape(n_obs,1)
        if noise == True:
            y = y + np.random.randn(n_obs).reshape(-1,1)*0.15 
        
    return X.astype('float32'), y.astype('float32')


# Modelling a two-Input Nonlinear Function (Sinc Equation)
def sinc_equation(x1,x2):
    return ((np.sin(x1)/x1) * (np.sin(x2)/x2))

def sinc_data(n_obs, multiplier=2, noise=False):
    X = (np.random.rand(n_obs, 2)-.5)*multiplier
    y = sinc_equation(X[:,0], X[:,1]).reshape(-1,1)
    if noise == True:
        y = y + np.random.randn(n_obs)*0.1
    return X.astype('float32'), y.astype('float32')


# Modelling a Three-Input Nonlinear Function (Sinc Equation)
def nonlin_equation(x,y,z):
    return ((1 + x**0.5 + 1/y + z**(-1.5))**2)

def nonlin_data(n_obs, multiplier=1, noise=False):
    X = np.random.rand(n_obs, 3)*multiplier + 1
    y = nonlin_equation(X[:,0], X[:,1], X[:,2]).reshape(-1,1)
    if noise == True:
        y = y + np.random.randn(n_obs)
    return X.astype('float32'), y.astype('float32')


# Generate a general input matrix X from time series
def gen_X_from_y(x, n_input=1, batch_size=16):    
    n_obs = len(x)-n_input
    n_obs = n_obs - n_obs % batch_size  # ensure a size that is n_obs % batch_size == 0

    data = np.zeros((n_obs, n_input+1))
    for t in range(n_input, n_obs + n_input):
        data[t - n_input,:] = [x[t-i] for i in range(n_input+1)]
        X = data[:,1:]
        y = data[:,0].reshape(n_obs,1)
        
    return X.astype('float32'), y.astype('float32')
