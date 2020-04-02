# -*- coding: utf-8 -*-
"""
Experiment for Anfis Parameters
"""
import myanfis
import numpy as np
import time
import data_sim as sim
import tensorflow as tf
import datetime
import os
from tensorflow.keras.callbacks import TensorBoard
##############################################################################
## Model Parameter
param = myanfis.fis_parameters(
            n_input = [2,3,5],                # no. of Regressors
            n_memb = [2,3,5],                 # no. of fuzzy memberships
            batch_size = 16,            # 16 / 32 / 64 / ...
            memb_func =   ['gaussian','bell'],     # 'gaussian' / 'bell'
            optimizer = ['sgd', 'adam'],         # sgd / adam / ...
            loss = ['mse', 'mae', 'huber_loss'],        # mse / mae / huber_loss / hinge / ...
            n_epochs = 25               # 10 / 25 / 50 / 100 / ...
            )      

## Data Parameters
data_set = 1                            # 1 = regression / 2 = mackey / 3 = sinc/ 
                                        # 4 = Three-Input Nonlin /5 = diabetes
n_obs = 2080

## General Parameters
core = '/device:CPU:0'                  # '/device:CPU:0' // '/device:GPU:0'
show_core_usage = False                 # True / False
# to call tensorboard type in prompt >> tensorboard --logdir=logs/sim_anfis
##############################################################################    

# Make ANFIS
tf.debugging.set_log_device_placement(show_core_usage) # find out which devices your operations and tensors are assigned to

n_exp = 0
start_time = time.time()

with tf.device(core):  # CPU / GPU
    for memb_func in param.memb_func:
        for optimizer in param.optimizer:
            for loss in param.loss:    
                for n_input in param.n_input:
                    for n_memb in param.n_memb:
                        # Generate Data
                        X, X_train, X_test, y, y_train, y_test = sim.gen_data(data_set, n_obs, n_input, n_memb)
                        
                        # set up tensorboard call back
                        log_name = f'-N{n_input}_M{n_memb}_{memb_func}_{optimizer}_{loss}'
                        path = os.path.join("logs", "exp_regression",
                                            datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
                                            + log_name
                                            )
                        tensorboard_callback = TensorBoard(log_dir=path, histogram_freq=1)

                        # create model
                        fis = myanfis.ANFIS(n_input = n_input, 
                                            n_memb = n_memb, 
                                            batch_size = param.batch_size, 
                                            memb_func = memb_func,
                                            name = 'myanfis'
                                            )
                        
                        # compile model
                        fis.model.compile(optimizer=optimizer, 
                                          loss=loss 
                                          ,metrics=['mae', 'mse']
                                          )
                        
                        # fit model
                        history = fis.fit(X_train, y_train, 
                                          epochs=param.n_epochs, 
                                          batch_size=param.batch_size,
                                          validation_data = (X_test, y_test),
                                          callbacks = [tensorboard_callback]
                                          )  
                        
                        n_exp += 1

end_time = time.time()
time = np.round(end_time - start_time,2)
print(f'Time for experiment: {np.round(end_time - start_time,2)} seconds')
print(f'number of experimental settings was {n_exp} ')
