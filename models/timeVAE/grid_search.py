# The grid_search.py file is a modification of test_vae.py.

import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # disabling gpu usage because my cuda is corrupted, needs to be fixed. 

import sys
import numpy as np , pandas as pd
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from vae_dense_model import VariationalAutoencoderDense as VAE_Dense
from vae_conv_model import VariationalAutoencoderConv as VAE_Conv
from vae_conv_I_model import VariationalAutoencoderConvInterpretable as TimeVAE
import utils
import argparse



if __name__ == '__main__':
    # Model complexity hyper-parameters:
    layer_sizes       = [[50, 100, 200], [75, 150, 300], [100, 200, 400]]
    latent_dims       = [4, 8, 12]
    trend_polynomials = [2, 4, 6]
    # Training hyper-parameters:
    learning_rates    = [1e-3, 1e-4, 1e-5]

    flag = 0

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--dataset_name', type=str, default = None)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    
    data_dir = '../datasets/'
    # ----------------------------------------------------------------------------------
    # choose model
    vae_type = 'timeVAE'           # vae_dense, vae_conv, timeVAE
    # ----------------------------------------------------------------------------------
    # read data    
    #valid_perc = 0.1
    train_input_file = f'{dataset_name}_train.npz'
    train_data = utils.get_training_data(data_dir + train_input_file)
    N_t, T_t, D_t = train_data.shape   
    print('train data shape:', N_t, T_t, D_t)

    valid_input_file = f'{dataset_name}_valid.npz'
    valid_data = utils.get_training_data(data_dir + valid_input_file)
    N_v, T_v, D_v = valid_data.shape   
    print('validation data shape:', N_v, T_v, D_v) 
    
    # ----------------------------------------------------------------------------------
    # min max scale the data    
    scaler = utils.MinMaxScaler()        
    scaled_train_data = scaler.fit_transform(train_data)

    flag = 0
    
    # Start the grid search over the selected hyper-parameters:
    for ls in layer_sizes:
        for ld in latent_dims:
            for tp in trend_polynomials:
                for lr in learning_rates:
                    # ----------------------------------------------------------------------------------
                    # Print the current model settings:
                    print("\n")
                    print("-------------------------------------------------------------------------------")
                    print(f"Now training VAE #{flag} with ls = {ls}, ld = {ld}, tp = {tp}, lr = {lr}")
                    print("-------------------------------------------------------------------------------")
                    print("\n") 

                    # Start the timer:
                    start = time.time()

                    # ----------------------------------------------------------------------------------
                    # instantiate the model     
                    if vae_type == 'vae_dense': 
                        vae = VAE_Dense( seq_len=T_t, feat_dim=D_t, latent_dim=ld, hidden_layer_sizes=ls, )
                    elif vae_type == 'vae_conv':
                        vae = VAE_Conv( seq_len=T_t, feat_dim=D_t, latent_dim=ld, hidden_layer_sizes=ls )
                    elif vae_type == 'timeVAE':
                        vae = TimeVAE( seq_len=T_t, feat_dim=D_t, latent_dim=ld, hidden_layer_sizes=ls,
                                reconstruction_wt = 3,
                                # ---------------------
                                # disable following three arguments to use the model as TimeVAE_Base. Enabling will convert to Interpretable version.
                                # Also set use_residual_conn= False if you want to only have interpretable components, and no residual (non-interpretable) component. 
                                
                                trend_poly=tp, 
                                custom_seas = [(6, 380)],     # list of tuples of (num_of_seasons, len_per_season)
                                # use_scaler = True,
                                
                                #---------------------------
                                use_residual_conn = True
                            )   
                    else:  raise Exception('wut')

                    
                    vae.compile(optimizer=Adam(learning_rate=lr))

                    early_stop_loss = 'loss'
                    early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-1, patience=10) 
                    reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5)

                    training_history = vae.fit(
                        scaled_train_data, 
                        batch_size = 32,
                        epochs=5000,
                        shuffle = True,
                        callbacks=[early_stop_callback, reduceLR],
                        verbose = 1
                    )
                    
                    # ----------------------------------------------------------------------------------    
                    # save model 
                    model_dir = './model/'
                    signiture = f"{dataset_name}_{ls}_{ld}_{tp}_{str(lr)}"
                    file_pref = f'{vae_type}_{signiture}'
                    vae.save(model_dir, file_pref)
                    
                    # ----------------------------------------------------------------------------------
                    # visually check reconstruction 
                    X = scaled_train_data

                    x_decoded = vae.predict(scaled_train_data)
                    print('x_decoded.shape', x_decoded.shape)                        
                    # # ----------------------------------------------------------------------------------
                    # draw random prior samples
                    print("num_samples: ", N_v)

                    samples = vae.get_prior_samples(num_samples=N_v)
                    
                    # Generate plots:
                    plot_path = "./outputs/grid_search/plots/"
                    utils.plot_samples(samples, n=5, path=plot_path, signiture=signiture)
                    utils.plot_losses(training_history, path=plot_path, signiture=signiture)
                    utils.draw_orig_and_post_pred_sample(X, x_decoded, n=5, path=plot_path, signiture=signiture)
                    
                    # Plot the prior generated samples over different areas of the latent space
                    if ld == 2: utils.plot_latent_space_timeseries(vae, n=8, figsize = (20, 10), path=plot_path, signiture=signiture)

                    # inverse-transform scaling 
                    samples = scaler.inverse_transform(samples)
                    print('shape of gen samples: ', samples.shape) 

                    # ----------------------------------------------------------------------------------
                    # Save the synthethic data for further analysis:
                    output_dir = './outputs/grid_search/data'
                    sample_fname = f'{signiture}_gen.npz'
                    samples_fpath = os.path.join(output_dir, sample_fname) 
                    np.savez_compressed(samples_fpath, data=samples)
                    np.savez_compressed("../TSGBench/data/timevae/grid_search/" + sample_fname, data=samples)

                    # ----------------------------------------------------------------------------------
                    
                    # later.... load model 
                    new_vae = TimeVAE.load(model_dir, file_pref)

                    new_x_decoded = new_vae.predict(scaled_train_data)
                    print('new_x_decoded.shape', new_x_decoded.shape)

                    print('Preds from orig and loaded models equal: ', np.allclose( x_decoded,  new_x_decoded, atol=1e-5))        
                    
                    # ----------------------------------------------------------------------------------
                    
                    # Stop the timer and print out the time:
                    end = time.time()
                    print(f"Total run time: {np.round((end - start)/60.0, 2)} minutes")

                    flag += 1