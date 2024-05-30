# The grid_search.py file is a modification of ICLR 2021 - Experiment 1.ipynb.

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

# Load the data
ori_data = np.load('../data/fmri_train.npz')
try:
    ori_data = ori_data['data']
except:
    ori_data = ori_data['arr_0']
print("ori_data shape:", ori_data.shape)

valid_data = np.load('../data/fmri_valid.npz')
try:
    valid_data = valid_data['data']
except:
    valid_data = valid_data['arr_0']
print("valid_data shape:", valid_data.shape)
n_valid, _, _ = valid_data.shape

# Restructure the data
rows, cols, dims = ori_data.shape
X = []
for i in range(rows):
    temp = []
    for j in range(cols):
        for k in range(dims):
            temp.append(ori_data[i, j, k])
    X.append(np.array(temp))

# Model complexity hyper-parameters:
layers_sizes   = [150, 200, 250]
n_flows        = [5, 10, 15]
# Training hyper-parameters:
learning_rates = [1e-3, 1e-4, 1e-5]

flag = 0

# Start the grid search over the selected hyper-parameters:
for ls in layers_sizes:
    for nf in n_flows:
        for lr in learning_rates:

            # ----------------------------------------------------------------------------------
            # Print the current model settings:
            print("\n")
            print("--------------------------------------------------------------------------")
            print(f"Now training Fourier-flows #{flag} with ls = {ls}, nf = {nf}, lr = {lr}")
            print("--------------------------------------------------------------------------")
            print("\n")

            # Start the timer:
            start = time.time()

            # Instantiate the model
            FF_model = FourierFlow(hidden=ls, fft_size=cols*dims, n_flows=nf, normalize=False, FFT=False)
            
            print(np.asarray(X).shape)

            # Fit the model
            FF_losses = FF_model.fit(X, epochs=5000, batch_size=128, learning_rate=lr, display_step=100)

            # Generate new data
            X_gen_FF = FF_model.sample(rows)

            # Reshape generated data
            gen_data = []
            for i in range(n_valid):
                temp = []
                for j in range(dims):
                    temp.append(X_gen_FF[i][j::dims])
                gen_data.append(temp)

            gen_data = np.array(gen_data)
            gen_data = np.transpose(gen_data, (0, 2, 1))
            print("gen data shape:", gen_data.shape)

            plot_path = f"./outputs/grid_search/plots/"
            signiture = f"fmri_{ls}_{nf}_{str(lr)}"

            # Generate plots:
            plot_losses(np.array(FF_losses), plot_path, signiture)

            # Save the synthethic data for further analysis:
            data_path = f"./outputs/grid_search/data/"
            np.savez(f'{data_path}{signiture}_gen.npz', gen_data)
            np.savez_compressed(f"../TSGBench/data/fourierflows/grid_search/{signiture}_gen.npz", gen_data)

            # Stop the timer and print out the time:
            end = time.time()
            print(f"Total run time: {np.round((end - start)/60.0, 2)} minutes")

            flag += 1
