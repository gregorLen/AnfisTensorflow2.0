
"""
Data Simulation for Anfis Sandbox
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from MS_generator import MRS
from STAR_generator import STAR
import matplotlib.pyplot as plt
plt.rcParams['axes.xmargin'] = 0 # remove margins from all plots
##############################################################################
def gen_data(data_id, n_obs, n_input, n_control, batch_size=16, lag=1):
     
    # Threshold Autoregressive Model (TAR)
    if data_id == 0 :
        mu_params =     np.array( [0.15, -0.08])
        sigma_params =  np.array( [0.0, 0.0])
        AR_params =     np.array([[0.0, 0 ],
                                  [0.0, 0] ])   
        gamma = float('inf')
        star_model = STAR(mu_params, sigma_params, AR_params, gamma)        
        star_model.sim(n_obs+n_input)
        star_model.plot(colored=True)
        X, y = gen_X_from_y(star_model.r, star_model.transvar, n_input, lag)
    
    # Smooth Transition Autoregressive Model (STAR)
    elif data_id == 1:
        mu_params =     np.array( [0.15, -0.10])
        sigma_params =  np.array( [0.0, 0.0])
        AR_params =     np.array([[0.0, 0 ],
                                  [0.0, 0] ])   
        gamma = 1
        star_model = STAR(mu_params, sigma_params, AR_params, gamma)        
        star_model.sim(n_obs+n_input)
        star_model.plot(colored=True)
        X, y = gen_X_from_y(star_model.r, star_model.transvar, n_input, lag)
    
    
    
    # standardize   
    scaler = StandardScaler()   
    X = scaler.fit_transform(X)
    
    scaler = StandardScaler()   
    y = scaler.fit_transform(y)
          
    # split data into test and train set
    X, X_train, X_test, y, y_train, y_test = split_data(X, y, batch_size)
    

    return X, X_train, X_test, y, y_train, y_test

##############################################################################
def get_data_name(data_id):
    data_sets = ['TAR time series', 'STAR time series']
    
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

# Generate a input matrix X from time series y
def gen_X_from_y(x, transvar, n_input=1, lag=1):    
    n_obs = len(x) - n_input*lag
    
    data = np.zeros((n_obs, n_input+1))
    for t in range(n_input*lag, n_obs + n_input*lag):
        data[t - n_input*lag,:] = [x[t-i*lag] for i in range(n_input+1)]
    X = np.concatenate((data[:,1:].reshape(n_obs,-1), transvar.reshape(-1,1)[n_input*lag:] ), axis=1)
    y = data[:,0].reshape(n_obs,1)
    
    return X.astype('float32'), y.astype('float32')























