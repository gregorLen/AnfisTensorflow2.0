"""
BASE SIMULATION MYANFIS (SANDBOX)
"""
import myanfis
import numpy as np
import time
import data_gen as gen
import tensorflow as tf
import datetime
import os
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
# import tensorflow.keras.optimizers as optimizers    # <-- for specifying optimizer
##############################################################################
## Model Parameter
param = myanfis.fis_parameters(
            n_input = 4,                # no. of Regressors
            n_memb = 3,                 # no. of fuzzy memberships
            batch_size = 16,            # 16 / 32 / 64 / ...
            memb_func = 'gaussian',     # 'gaussian' / 'bell'
            optimizer = 'adam',          # sgd / adam / ...
            loss = 'huber_loss',        # mse / mae / huber_loss / hinge / ...
            n_epochs = 20               # 10 / 25 / 50 / 100 / ...
            )      

## Data Parameters
n_obs = 2080
data_set = 2                            # 1 = markov regime switching ts / 
                                        # 2 = mackey / 3 = sinc/ 
                                        # 4 = Three-Input Nonlin /5 = diabetes / 
                                        # 6 = artificial regression
## General Parameters
plot_learningcurves = True              # True / False
plot_mfs = True                         # True / False
plot_heatmap = True                     # True / False
show_summary = True                     # True / False
core = '/device:CPU:0'                  # '/device:CPU:0' // '/device:GPU:0'
show_core_usage = False                 # True / False
##############################################################################    
# Generate Data
X, X_train, X_test, y, y_train, y_test = gen.gen_data(data_set, n_obs, param.n_input, param.batch_size)

# show which devices your operations are assigned to
tf.debugging.set_log_device_placement(show_core_usage) 

with tf.device(core):  # CPU / GPU
    # set tensorboard call back
    log_name = f'-data_{data_set}_N{param.n_input}_M{param.n_memb}_batch{param.batch_size}_{param.memb_func}_{param.optimizer}_{param.loss}'
    log_path = os.path.join("logs", "run_anfis",
                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
                        + log_name
                        )
    tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1)
    
    # create model
    fis = myanfis.ANFIS(n_input = param.n_input, 
                        n_memb = param.n_memb, 
                        batch_size = param.batch_size, 
                        memb_func = param.memb_func,
                        name = 'myanfis'
                        )
    
    # compile model
    fis.model.compile(optimizer=param.optimizer, 
                      loss=param.loss 
                      ,metrics=['mse']  # ['mae', 'mse']
                      )
    
    # fit model
    start_time = time.time()
    history = fis.fit(X_train, y_train, 
                      epochs=param.n_epochs, 
                      batch_size=param.batch_size,
                      validation_data = (X_test, y_test),
                      callbacks = [tensorboard_callback]
                      )  
    end_time = time.time()
    print(f'Time to fit: {np.round(end_time - start_time,2)} seconds')
    

# ## Evaluate Model
# fis.model.evaluate(X_test, y_test)  
if plot_mfs:
    fis.plotmfs()

if plot_learningcurves:
    loss_curves = pd.DataFrame(history.history)
    loss_curves.plot(figsize=(8, 5))
    plt.grid(True)
    plt.show()

memberships = fis.get_memberships(X)    
y_pred = fis.model.predict(X)

f, axs = plt.subplots(3,1,figsize=(8,15))
plt.subplot(3,1,1)
plt.plot(y)
plt.plot(y_pred, alpha=.5)
plt.legend(['Real', 'Predicted'])
plt.subplot(3,1,2)
plt.plot(np.arange(y.shape[0]), y - y_pred)
plt.legend(['pred_error'])
plt.subplot(3,1,3)
sns.heatmap(memberships.T, fmt="f", xticklabels=200, yticklabels=False,cbar_kws={"orientation": "horizontal"},
            vmin = memberships.min(), vmax=memberships.max(),
            cmap=None)  # twilight_shifted
#plt.stackplot(np.arange(memberships.shape[0]),memberships.T)  # alternative
plt.show()

        
if show_summary:
    print(fis.model.summary())


