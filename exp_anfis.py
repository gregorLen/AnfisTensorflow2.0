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
from tensorboard.plugins.hparams import api as hp
##############################################################################
## Model (Hyper) Parameter
HP_n_input = hp.HParam('n_ipnut', hp.Discrete([2, 3, 5]))                   # no. of Regressors
HP_n_memb = hp.HParam('n_memb', hp.Discrete([2, 3, 5]))                     # no. of fuzzy memberships
HP_memb_func = hp.HParam('memb_func', hp.Discrete(['gaussian', 'bell']))     # 'gaussian' / 'bell'
HP_optimizer = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))         # sgd / adam / ...
HP_loss = hp.HParam('loss', hp.Discrete(['mse', 'mae', 'huber_loss']))      # mse / mae / huber_loss / hinge / ...

METRIC = 'mse'

batch_size = 16 
n_epochs = 25     

## Data Parameters
n_obs = 2080
data_set = 2                            # 1 = regression / 2 = mackey / 3 = sinc/ 
                                        # 4 = Three-Input Nonlin /5 = diabetes

## General Parameters
core = '/device:CPU:0'                  # '/device:CPU:0' // '/device:GPU:0'
show_core_usage = False                 # True / False
# to call tensorboard type in prompt >> tensorboard --logdir=logs/sim_anfis
##############################################################################    
# RUN ANFIS
tf.debugging.set_log_device_placement(show_core_usage) # find out which devices your operations and tensors are assigned to

with tf.summary.create_file_writer('logs/exp_anfis').as_default():
    hp.hparams_config(hparams=[HP_n_input, HP_n_memb, HP_memb_func, HP_optimizer, HP_loss],
                      metrics=[hp.Metric(METRIC, display_name='mse')]
                      )


def train_test_model(hparams):
    # Generate Data
    X, X_train, X_test, y, y_train, y_test = sim.gen_data(data_set, n_obs, hparams[HP_n_input], hparams[HP_n_memb])
    
    # create model
    fis = myanfis.ANFIS(n_input = hparams[HP_n_input], 
                        n_memb = hparams[HP_n_memb], 
                        batch_size = batch_size, 
                        memb_func = hparams[HP_memb_func],
                        name = 'myanfis'
                        )
    
    # compile model
    fis.model.compile(optimizer=hparams[HP_optimizer], 
                      loss=hparams[HP_loss], 
                      metrics=['mse'],
                      )
    # fit model
    fis.fit(X_train, y_train, 
            epochs=n_epochs, 
            batch_size=batch_size#,
            #validation_data = (X_test, y_test),
            # callbacks = [TensorBoard(logdir),  # log metrics
            #              hp.KerasCallback(logdir, hparams),  # log hparams
            #              ]
            )  
    _, evaluation = fis.model.evaluate(X_test, y_test)

    return evaluation


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        evaluation = train_test_model(hparams)
        tf.summary.scalar(METRIC, evaluation, step=1)




session_num = 0

for n_input in HP_n_input.domain.values:
    for n_memb in HP_n_memb.domain.values:
        for memb_func in HP_memb_func.domain.values:
            for optimizer in HP_optimizer.domain.values:
                for loss in HP_loss.domain.values:

                  hparams = {HP_n_input: n_input,
                             HP_n_memb: n_memb,
                             HP_memb_func: memb_func,
                             HP_optimizer: optimizer,
                             HP_loss: loss
                             }
                  
                  run_name = f'run-{session_num}'  
                  print(f'--- Starting trial: {run_name}')
                  print({h.name: hparams[h] for h in hparams})
                  run('logs/exp_anfis/' + run_name, hparams)
                  session_num += 1



# n_exp = 0
# start_time = time.time()

# with tf.device(core):  # CPU / GPU
#     for memb_func in param.memb_func:
#         for optimizer in param.optimizer:
#             for loss in param.loss:    
#                 for n_input in param.n_input:
#                     for n_memb in param.n_memb:
#                         # Generate Data
#                         X, X_train, X_test, y, y_train, y_test = sim.gen_data(data_set, n_obs, n_input, n_memb)
                        
#                         # set up tensorboard call back
#                         log_name = f'-N{n_input}_M{n_memb}_{memb_func}_{optimizer}_{loss}'
#                         path = os.path.join("logs", "exp_regression",
#                                             datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
#                                             + log_name
#                                             )
#                         tensorboard_callback = TensorBoard(log_dir=path, histogram_freq=1)

#                         # create model
#                         fis = myanfis.ANFIS(n_input = n_input, 
#                                             n_memb = n_memb, 
#                                             batch_size = param.batch_size, 
#                                             memb_func = memb_func,
#                                             name = 'myanfis'
#                                             )
                        
#                         # compile model
#                         fis.model.compile(optimizer=optimizer, 
#                                           loss=loss 
#                                           ,metrics=['mae', 'mse']
#                                           )
                        
#                         # fit model
#                         history = fis.fit(X_train, y_train, 
#                                           epochs=param.n_epochs, 
#                                           batch_size=param.batch_size,
#                                           validation_data = (X_test, y_test),
#                                           callbacks = [tensorboard_callback]
#                                           )  
                        
#                         n_exp += 1

# end_time = time.time()
# time = np.round(end_time - start_time,2)
# print(f'Time for experiment: {np.round(end_time - start_time,2)} seconds')
# print(f'number of experimental settings was {n_exp} ')
