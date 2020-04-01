"""
BASE SIMULATION MYANFIS
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
##############################################################################
## Model Parameter
opti = optimizers.SGD(learning_rate=0.2, momentum=0.01, nesterov=False)

param = myanfis.fis_parameters(
            n_input = 3,                # no. of Regressors
            n_memb = 4,                 # no. of fuzzy memberships
            batch_size = 16,            # 16 / 32 / 64 / ...
            memb_func = 'gaussian',     # 'gaussian' / 'bell'
            optimizer = 'adam',         # sgd / adam / ...
            loss = 'huber_loss',        # mse / mae / huber_loss / hinge / ...
            n_epochs = 25               # 10 / 25 / 50 / 100 / ...
            )      

## Data Parameters
data_set = 3                            # 1 = random / 2 = mackey / 3 = sinc/ 
                                        # 4 = Three-Input Nonlin /5 = diabetes
n_obs = param.batch_size * 100

## General Parameters
plot_learningcurves = False             # True / False
plot_mfs = True                         # True / False
show_summary = True                     # True / False

core = '/device:CPU:0'                  # '/device:CPU:0' // '/device:GPU:0'
show_core_usage = False                 # True / False
    # set up tensorboard call back
log_name = f'-epoch{param.n_epochs}_N{param.n_input}_M{param.n_memb}_{param.optimizer}_{param.loss}'
path = os.path.join("logs", "sim_anfis",
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
                    + log_name
                    )
tensorboard_callback = TensorBoard(log_dir=path, histogram_freq=1)
# to call tensorboard type in prompt >> tensorboard --logdir=logs/sim_anfis
##############################################################################    
# Generate Data
X, X_train, X_test, y, y_train, y_test = sim.gen_data(data_set, n_obs, param.n_input, param.n_memb)

# Make ANFIS
tf.debugging.set_log_device_placement(show_core_usage) # find out which devices your operations and tensors are assigned to

with tf.device(core):  # CPU / GPU
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
    time = np.round(end_time - start_time,2)
    print(f'Time to fit: {np.round(end_time - start_time,2)} seconds')
    

# ## Evaluate Model
# fis.model.evaluate(X_test, y_test)  
if plot_learningcurves:
    loss_curves = pd.DataFrame(history.history)
    loss_curves.plot(figsize=(8, 5))
    plt.grid(True)
    plt.show()
    
y_pred = fis.model.predict(X)
if data_set == 2 or 3:
    plt.subplot(2,1,1)
    plt.plot(y)
    plt.plot(y_pred)
    plt.legend(['Real', 'Predicted'])
    plt.subplot(2,1,2)
    plt.plot(np.arange(y.shape[0]), y - y_pred)
    plt.legend(['pred_error'])
    plt.show()
    
if plot_mfs:
    fis.plotmfs()
    
if show_summary:
    print(fis.model.summary())

#state_similarity = fis.get_state_similarity(X)
#states = np.argmax(state_similarity, axis=1)
#pd.crosstab(states, columns='count')

