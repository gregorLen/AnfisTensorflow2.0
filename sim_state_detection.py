"""
MRS Sim State detection
Gregor Lenhard
University of Basel
"""
##############################################################################
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import myanfis
import datetime
import seaborn as sns
import os
from markovstate_generator import MRS
from data_sim import gen_X_from_y
import numpy as np
import time
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
##############################################################################
# Markov Process Parameters
P = np.array([          [0.989, 0.01,   0.001],        ## Transition Matrix
                        [0.03,  0.969,  0.001], 
                        [0.00,  0.03,   0.97] ])

mu_params = np.array(   [0.08,0.0,-0.60])
 
sigma_params = np.array([.1,.25,.60])
 
AR_params = np.array([  [0.4, -0.2],
                        [0.5, -0.3],
                        [0.8, -.4]])  


# Model Parameters
#opti = optimizers.SGD(learning_rate=0.02, momentum=0.01, nesterov=False)
param = myanfis.fis_parameters(
            n_input = 2,  # no. of Regressors
            n_memb = 3,  # no. of fuzzy memberships
            batch_size = 16,
            memb_func = 'gaussian',  # 'gaussian' / 'bell'
            optimizer = 'nadam',   # sgd / adam / 
            loss = 'mse',     # mse / mae / huber_loss / hinge
            n_epochs = 30
            )  

# data parameters
n_obs = 2000

## General Parameters
plot_learningcurves = True              # True / False
plot_mfs = True                         # True / False
show_summary = True                     # True / False
core = '/device:CPU:0'                  # '/device:CPU:0' // '/device:GPU:0'
##############################################################################
# Generate Data
#np.random.seed(0)
mrs_model = MRS(P, mu_params, sigma_params, AR_params)
mrs_model.sim(n_obs)
mrs_model.plot_sim(colored=True)
X, y = gen_X_from_y(mrs_model.r, param.n_input, param.batch_size)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.29)    
##############################################################################
# Run Model
with tf.device(core):  # CPU / GPU
    
    # set tensorboard call back
    log_name = f'-N{param.n_input}_M{param.n_memb}_batch{param.batch_size}_{param.memb_func}_{param.optimizer}_{param.loss}'
    log_path = os.path.join("logs", "sim_state_detection",
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
    
    # Fit Model
    start_time = time.time()
    history = fis.fit(X_train, y_train, 
                      epochs=param.n_epochs, 
                      batch_size=param.batch_size,
                      validation_data = (X_test, y_test),
                      callbacks = [tensorboard_callback]
                      ) 
    end_time = time.time()
    print(f'Time to fit: {np.round(end_time - start_time,2)} seconds')

# ############################################################################
# # Evaluate Model
y_pred = fis.model.predict(X)
state_similarity = fis.get_state_similarity(X)
states = np.argmax(state_similarity, axis=1)
state_distribution = pd.crosstab(states, columns='count')

# plot predictions and state heatmap
f, axs = plt.subplots(2,1,figsize=(8,15))
plt.subplot(2,1,1)
plt.plot(y)
plt.plot(y_pred, alpha=.5)
plt.margins(x=0)
plt.legend(['Real', 'Predicted'])
# plt.subplot(2,1,2)
# plt.plot(np.arange(len(y)), y - y_pred)
# plt.margins(x=0)
# plt.legend(['pred_error'])
plt.subplot(2,1,2)
df_states = pd.DataFrame(state_similarity)
plt.stackplot(np.arange(df_states.shape[0]),df_states.T)
# sns.heatmap(state_similarity.T, fmt="f", xticklabels=250,cbar_kws={"orientation": "horizontal"},
#         vmin = state_similarity.min(), vmax=state_similarity.max(),
#         cmap=None)  # twilight_shifted
plt.show()

# plot learning curves
if plot_learningcurves:
    loss_curves = pd.DataFrame(history.history)
    loss_curves.plot(figsize=(8, 5))
    plt.grid(True)
    plt.show()

# plot membership functions
if plot_mfs:
    fis.plotmfs()
    
if show_summary:
    print(fis.model.summary())

sns.heatmap(state_similarity.T, fmt="f", xticklabels=250,cbar_kws={"orientation": "horizontal"},
        vmin = state_similarity.min(), vmax=state_similarity.max(),
        cmap=None)  # twilight_shifted
#sns.pairplot(df_states)
