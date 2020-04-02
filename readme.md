#### Creating an ANFIS-based early waring system

## Using Tensoboard

# Step 1: Set working directory : 
>> cd /d S:\...\MyAnfis

# Step 2: Activate virtual environment
>> conda activate tensorflow

# Step 3: Start Tensorflow
for simanfis sandbox
>> tensorboard --logdir=logs/sim_anfis

for RegimeSwitching experiments
>> tensorboard --logdir=logs/sim_MRS

# Step 4: open Browser
localhost:6006