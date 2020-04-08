# ANFIS: Adaptive-Network-Based Fuzzy Inference System 
#####  A flexible implementation based on Tensorflow 2.0 and Keras
---
### Python implementation
The implementation is built around Tensorflow 2.0. 

```python
import myanfis
import data_sim as sim
import tensorflow as tf

param_obj = myanfis.fis_parameters()    # standard parameters

X, X_train, X_test, y, y_train, y_test = sim.gen_data(data_set=1,   # regression-type
                                                        n_obs=2080, 
                                                        param.n_input )
                                                        
fis = myanfis.ANFIS(n_input = param.n_input, 
                    n_memb = param.n_memb, 
                    batch_size = param.batch_size, 
                    memb_func = param.memb_func,
                    name = 'myanfis' )

fis.model.compile(optimizer=param.optimizer, 
                  loss=param.loss 
                  ,metrics=['mae', 'mse'] )

history = fis.fit(X_train, y_train, 
                  epochs=param.n_epochs, 
                  batch_size=param.batch_size,
                  validation_data = (X_test, y_test) )  
```
---
### Using Tensoboard
#### Step 0: Define a callback
```python
log_name = f'-data_{data_set}_N{param.n_input}_M{param.n_memb}_batch{param.batch_size}_{param.memb_func}_{param.optimizer}_{param.loss}'

log_path = os.path.join("logs", 
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + log_name )

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)

history = fis.fit(...
                  callbacks = [tensorboard_callback] )  

```



#### Step 1: Set working directory : 
$ cd /d S:\...\MyAnfis

#### Step 2: Activate virtual environment
$ conda activate tensorflow

#### Step 3: Start Tensorflow
for simanfis sandbox
$ tensorboard --logdir=logs/sim_anfis

for RegimeSwitching experiments
$ tensorboard --logdir=logs/sim_MRS

#### Step 4: open Browser
>> localhost:6006

---