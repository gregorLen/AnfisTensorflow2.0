
"""
Data Simulation for Anfis Sandbox
"""
import numpy as np
from sklearn.datasets import load_diabetes 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from markovstate_generator import MRS
import matplotlib.pyplot as plt
plt.rcParams['axes.xmargin'] = 0 # remove margins from all plots
##############################################################################
def get_data_name(data_id):
    data_sets = ['markovRS', 'mackey', 'sinc', 'threeInputNonlin', 'diabetes', 'regressoin']
    
    return data_sets[data_id]


def split_data(X, y, batch_size):
    # adjust X and y for batch_size
    adj_id = np.arange(len(y) - len(y)%batch_size)
    X, y = X[adj_id, :], y[adj_id]
    
    # split test & train according to batches
    batches = len(y) / batch_size
    train_batches = np.round(batches*0.6) / batches
    
    train_id = np.arange(len(y)*train_batches, dtype=int)
    test_id = np.arange(len(y)*train_batches, len(y), dtype=int)
    
    X_train, y_train, X_test, y_test = X[train_id,:], y[train_id], X[test_id,:], y[test_id]

    return X, X_train, X_test, y, y_train, y_test


def gen_data(data_id, n_obs, n_input, batch_size=16, lag=1):
    
    # Markov Regime switching ts
    if data_id == 0:  
        np.random.seed(121)       # set a seed for reproducable results
        mrs_model = MRS(P = np.array([[0.985, 0.01,   0.004],        
                                      [0.03,  0.969,  0.001], 
                                      [0.00,  0.03,   0.97] ]))
        
        mrs_model.sim(n_obs+n_input)
        mrs_model.plot_sim(colored=True)
        X, y = gen_X_from_y(mrs_model.r, n_input, lag)
        
    # Mackey    
    elif data_id == 1: 
        y = mackey(124+n_obs+n_input)[124:]
        X, y = gen_X_from_y(y, n_input, lag)
   
    # Nonlin sinc equation             
    elif data_id == 2:  
        X, y = sinc_data(n_obs)
        assert n_input == 2, 'Nonlin sinc equation data set requires n_input==2. Please chhange to 2.'

    # Nonlin three-input equation    
    elif data_id == 3:  
        X, y = nonlin_data(n_obs)
        assert n_input == 3, 'Nonlin Three-Input Equation required n_input==3. Please switch to 3.'
        
    # diabetes dataset from sklean
    elif data_id == 4: 
        n_obs = 400
        print('Dataset diabetes is limited to 400 observations')
        X, y = load_diabetes(return_X_y=True)
        X, y = X[:n_obs,0:n_input].astype('float32'), y[:n_obs].astype('float32').reshape(-1,1)

    # artificial regression-type
    elif data_id == 5: 
        X, y = make_regression(n_samples=n_obs, 
                               n_features=n_input, 
                               n_informative=n_input, 
                               n_targets=1, 
                               noise= 20     
                               )
        X, y = X.astype('float32'), y.astype('float32').reshape(-1,1)
    
    # standardize   
    scaler = StandardScaler()   
    X = scaler.fit_transform(X)
          
    # split data into test and train set
    X, X_train, X_test, y, y_train, y_test = split_data(X, y, batch_size)
    
    
    # alternative: shuffle & split
    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4) 
    # X_train, X_test, y_train, y_test = adjust_for_batch_size(X_train, X_test, y_train, y_test, batch_size)
    
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


# Generate a input matrix X from time series y
def gen_X_from_y(x, n_input=1, lag=1):    
    n_obs = len(x) - n_input*lag
    
    data = np.zeros((n_obs, n_input+1))
    for t in range(n_input*lag, n_obs + n_input*lag):
        data[t - n_input*lag,:] = [x[t-i*lag] for i in range(n_input+1)]
    X = data[:,1:].reshape(n_obs,-1)
    y = data[:,0].reshape(n_obs,1)
    
    return X.astype('float32'), y.astype('float32')























