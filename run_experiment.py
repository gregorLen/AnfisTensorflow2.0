"""
Experiment for Anfis Sandbox
"""
from Models import myanfis
import numpy as np
import time
import Datagenerator.datagenerator as gen
import tensorflow as tf
import datetime
import os
from tensorflow.keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import api as hp
##############################################################################
# Initializing HyperParameters
n_input = [2, 3, 5]
HP_n_input = hp.HParam('n_ipnut', hp.Discrete(
    n_input))                         # no. of Regressors
# no. of fuzzy memberships
HP_n_memb = hp.HParam('n_memb', hp.Discrete([2, 3, 4, 5]))
HP_memb_func = hp.HParam('memb_func', hp.Discrete(
    ['gaussian', 'gbellmf']))        # 'gaussian' / 'bell'
# sgd / adam / ...
HP_optimizer = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
# mse / mae / huber_loss / hinge / ...
HP_loss = hp.HParam('loss', hp.Discrete(['mse', 'mae', 'huber_loss']))
METRIC = 'mse'

# Model Parameters
batch_size = 16
n_epochs = 50

# Data Parameters
n_obs = 2000
lag = 1
data_id = 1                             # 0 = mackey / 1 = sinc /
# 2 = Three-Input Nonlin /
# 3 = markov switching
# 4 = TAR  /  # 5 = STAR

# General Parameters
core = '/device:CPU:0'                  # '/device:CPU:0' // '/device:GPU:0'
show_core_usage = False                 # True / False
##############################################################################
# find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(show_core_usage)

# create logs for the training process that records the losses
with tf.summary.create_file_writer('logs/exp_anfis').as_default():
    hp.hparams_config(hparams=[HP_n_input, HP_n_memb, HP_memb_func, HP_optimizer, HP_loss],
                      metrics=[hp.Metric(METRIC, display_name='mse')]
                      )

# A function To Train And Validate


def train_test_model(logdir, hparams):
    # create model
    fis = myanfis.ANFIS(n_input=hparams[HP_n_input],
                        n_memb=hparams[HP_n_memb],
                        batch_size=batch_size,
                        memb_func=hparams[HP_memb_func],
                        name='myanfis'
                        )

    # compile model
    fis.model.compile(optimizer=hparams[HP_optimizer],
                      loss=hparams[HP_loss],
                      metrics=['mse'],             # TODO: add ['mae', 'mse']
                      )

    # fit model
    fis.fit(X_train[:, :hparams[HP_n_input]], y_train,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_data=(X_test[:, :hparams[HP_n_input]], y_test),
            callbacks=[TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch=100000000),  # log metrics
                       hp.KerasCallback(logdir, hparams),  # log hparams
                       ]
            )

    _, evaluation = fis.model.evaluate(X_test[:, :hparams[HP_n_input]], y_test)

    return evaluation

# A function to log the training process


def run(logdir, hparams):
    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        evaluation = train_test_model(logdir, hparams)
        tf.summary.scalar(METRIC, evaluation, step=1)


# Generate Data (same for every session)
X, X_train, X_test, y, y_train, y_test = gen.gen_data(
    data_id, n_obs, max(n_input), batch_size, lag)

# Start experiment
start_time = time.time()
session_num = 0
with tf.device(core):  # CPU / GPU
    for n_input in HP_n_input.domain.values:
        for n_memb in HP_n_memb.domain.values:
            for memb_func in HP_memb_func.domain.values:
                for optimizer in HP_optimizer.domain.values:
                    for loss in HP_loss.domain.values:

                        # generate hyperparameters
                        hparams = {HP_n_input: n_input,
                                   HP_n_memb: n_memb,
                                   HP_memb_func: memb_func,
                                   HP_optimizer: optimizer,
                                   HP_loss: loss
                                   }

                        # generate log path
                        session_name = f'-{gen.get_data_name(data_id)}_N{n_input}_M{n_memb}_{memb_func}_{optimizer}_{loss}'
                        logdir = os.path.join("logs", "exp_anfis",
                                              datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                                              + session_name
                                              )

                        # run session
                        print(f'--- Starting trial: {session_num}')
                        print({h.name: hparams[h] for h in hparams})
                        run(logdir, hparams)
                        session_num += 1

end_time = time.time()
time = np.round(end_time - start_time, 2)
print(f'Time for experiment: {np.round(end_time - start_time,2)} seconds')
print(f'number of experimental settings was {session_num} ')
