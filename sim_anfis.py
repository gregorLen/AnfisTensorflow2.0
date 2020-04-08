"""
BASE SIMULATION MYANFIS (SANDBOX)
"""
import myanfis
import numpy as np
import time
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
import data_sim as sim
import tensorflow as tf
import datetime
import os
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
##############################################################################
## Model Parameter
opti = optimizers.SGD(learning_rate=0.2, momentum=0.01, nesterov=False)

param = myanfis.fis_parameters(
            n_input = 2,                # no. of Regressors
            n_memb = 2,                 # no. of fuzzy memberships
            batch_size = 16,            # 16 / 32 / 64 / ...
            memb_func = 'gaussian',     # 'gaussian' / 'bell'
            optimizer = 'adam',         # sgd / adam / ...
            loss = 'huber_loss',        # mse / mae / huber_loss / hinge / ...
            n_epochs = 30               # 10 / 25 / 50 / 100 / ...
            )      

## Data Parameters
n_obs = param.batch_size * 100
data_set = 1                           # 1 = regression / 2 = mackey / 3 = sinc/ 
                                        # 4 = Three-Input Nonlin /5 = diabetes
## General Parameters
plot_learningcurves = True              # True / False
plot_mfs = True                         # True / False
plot_heatmap =True                      # True / False
show_summary = True                     # True / False
core = '/device:CPU:0'                  # '/device:CPU:0' // '/device:GPU:0'
show_core_usage = False                 # True / False
##############################################################################    
# Generate Data
X, X_train, X_test, y, y_train, y_test = sim.gen_data(data_set, n_obs, param.n_input)

# Make ANFIS
tf.debugging.set_log_device_placement(show_core_usage) # find out which devices your operations and tensors are assigned to

with tf.device(core):  # CPU / GPU
    
    # set tensorboard call back
    log_name = f'-data_{data_set}_N{param.n_input}_M{param.n_memb}_batch{param.batch_size}_{param.memb_func}_{param.optimizer}_{param.loss}'
    log_path = os.path.join("logs", "sim_anfis",
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
                      ,metrics=['mae', 'mse']
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
    
y_pred = fis.model.predict(X)
if data_set == 1 or 2 or 3:
    plt.subplot(2,1,1)
    plt.plot(y)
    plt.plot(y_pred, alpha=.5)
    plt.legend(['Real', 'Predicted'])
    plt.subplot(2,1,2)
    plt.plot(np.arange(y.shape[0]), y - y_pred)
    plt.legend(['pred_error'])
    plt.show()
    
if plot_heatmap:
    state_similarity = fis.get_state_similarity(X)
    sns.heatmap(state_similarity.T, fmt="f", xticklabels=200, yticklabels=False,cbar_kws={"orientation": "horizontal"},
            vmin = state_similarity.min(), vmax=state_similarity.max(),
            cmap=None)  # twilight_shifted
    states = np.argmax(state_similarity, axis=1)
    state_distribution = pd.crosstab(states, columns='count')
    
if show_summary:
    print(fis.model.summary())


