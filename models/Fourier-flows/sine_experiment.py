# The sine_experiment.py file is a modification of ICLR 2021 - Experiment 1.ipynb.

from SequentialFlows import FourierFlow
import os
import random
import time
from metrics.visualization_metrics import *

import numpy as np
import torch
import tensorflow as tf


# THIS FILE IS BASICALLY A REWRITE OF 'ICLR 2021 - Epxeriment 1.ipynb'

# Set seeds
SEED = 12345
np.random.seed(SEED)
tf.random.set_random_seed(1234)
random.seed(SEED)
torch.manual_seed(SEED)

# TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

dataset_name = 'custom_380_sines_6w_correlated_final'

# Load the data
ori_data = np.load(f'../DataVault/{dataset_name}_train.npz')
try:
    ori_data = ori_data['data']
except:
    ori_data = ori_data['arr_0']
print("ori_data shape:", ori_data.shape)

test_data = np.load(f'../DataVault/{dataset_name}_test.npz')
try:
    test_data = test_data['data']
except:
    test_data = test_data['arr_0']
print("test_data shape:", test_data.shape)
n_test, _, _ = test_data.shape

# Restructure the data
rows, cols, dims = ori_data.shape
X = []
for i in range(rows):
    temp = []
    for j in range(cols):
        for k in range(dims):
            temp.append(ori_data[i, j, k])
    X.append(np.array(temp))

ls = 200
nf = 10
lr = 1e-3

# ----------------------------------------------------------------------------------
# Print the current model settings:
print("\n")
print("--------------------------------------------------------------------------")
print(f"Now training Fourier-flows with ls = {ls}, nf = {nf}, lr = {lr}")
print("--------------------------------------------------------------------------")
print("\n")

# Start the timer:
start = time.time()

# Instantiate the model
FF_model = FourierFlow(hidden=ls, fft_size=cols*dims, n_flows=nf, normalize=False, FFT=False)

print(np.asarray(X).shape)

# Fit the model
FF_losses = FF_model.fit(X, epochs=1000, batch_size=32, learning_rate=lr, display_step=100)

# Generate new data
X_gen_FF = FF_model.sample(rows)

# Reshape generated data
gen_data = []
for i in range(n_test):
    temp = []
    for j in range(dims):
        temp.append(X_gen_FF[i][j::dims])
    gen_data.append(temp)

gen_data = np.array(gen_data)
gen_data = np.transpose(gen_data, (0, 2, 1))
print("gen data shape:", gen_data.shape)

plot_path = f"./outputs/sine_experiment/plots/"
signiture = f"{dataset_name}_{ls}_{nf}_{lr}"

# Generate plots:
plot_losses(np.array(FF_losses), plot_path, signiture)

# Save the synthethic data for further analysis:
data_path = f"./outputs/sine_experiment/data/"
np.savez(f'{data_path}{signiture}_gen.npz', gen_data)
#np.savez_compressed(f"../TSGBench/data/fourierflows/sine_experiment/{signiture}_gen.npz", gen_data)

# Stop the timer and print out the time:
end = time.time()
print(f"Total run time: {np.round((end - start)/60.0, 2)} minutes")
