"""
MRS Sim State detection
Gregor Lenhard
University of Basel
"""
##############################################################################
import myanfis
from MarkovRS import MRS
from data_sim import gen_X_from_y
import numpy as np
import time
##############################################################################
# Data Parameters
N = 1500

# Model Parameters
#opti = optimizers.SGD(learning_rate=0.02, momentum=0.01, nesterov=False)
param = myanfis.fis_parameters(
            n_input = 2,  # no. of Regressors
            n_memb = 3,  # no. of fuzzy memberships
            batch_size = 16,
            memb_func = 'gaussian',  # 'gaussian' / 'bell'
            optimizer = 'adam',   # sgd / adam / 
            loss = 'mse',     # mse / mae / huber_loss / hinge
            n_epochs = 25 
            )      

##############################################################################
# Generate Data
np.random.seed(0)
mrs_model = MRS()
mrs_model.sim(N)
mrs_model.plot_sim(colored=True)
X, y = gen_X_from_y(mrs_model.r, param.n_input, param.batch_size)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.29)    

##############################################################################
# Build Anfis
fis = myanfis.ANFIS(param.n_input, param.n_memb, param.batch_size, param.memb_func)
fis.model.compile(optimizer = param.optimizer, loss = param.loss)

# Fit Model
start_time = time.time()
history = fis.fit(X_train, y_train, 
                  epochs=param.n_epochs, 
                  batch_size=param.batch_size,
                  validation_data = (X_test, y_test)
                  ) 
end_time = time.time()
print(f'Time to fit: {np.round(end_time - start_time,2)} seconds')

# ##############################################################################
# # Evaluate Model
import pandas as pd 
import matplotlib.pyplot as plt
loss_curves = pd.DataFrame(history.history)
loss_curves.plot(figsize=(8, 5))
plt.grid(True)
plt.show()

y_pred = fis.model.predict(X)

plt.subplot(2,1,1)
plt.plot(y)
plt.plot(y_pred)
plt.legend(['Real', 'Predicted'])
plt.subplot(2,1,2)
plt.plot(np.arange(len(y)), y - y_pred)
plt.legend(['pred_error'])
plt.show()

fis.plotmfs()


state_similarity = fis.get_state_similarity(X)
states = np.argmax(state_similarity, axis=1)
pd.crosstab(states, columns='count')

