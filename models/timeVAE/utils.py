import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import sys
import tensorflow as tf
from config import config as cfg
from sklearn.manifold import TSNE

TITLE_FONT_SIZE = 16

def get_training_data(input_file):
    loaded = np.load(input_file)

    # Resolve an inconsistancy we encountered
    # with Mac and Windows generated .npz files:
    try:
        return loaded['data']
    except:
        return loaded['arr_0']



def get_daily_data():
    data = pd.read_parquet(cfg.DATA_FILE_PATH_AND_NAME)
    data.rename(columns={ 'queueid': 'seriesid', 'date': 'ts', 'callvolume': 'v',}, inplace=True)
    data['ts'] = pd.to_datetime(data['ts'])
    data = data[['seriesid', 'ts', 'v']]
    return data



def get_mnist_data():
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    # mnist_digits = np.concatenate([x_train, x_test], axis=0)
    # mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    mnist_digits = x_train.astype("float32") / 255
    return mnist_digits



def draw_orig_and_post_pred_sample(orig, reconst, n, path, signiture):

    fig, axs = plt.subplots(n, 2, figsize=(10,6))
    i = 1
    for _ in range(n):
        rnd_idx = np.random.choice(len(orig))
        o = orig[rnd_idx]
        r = reconst[rnd_idx]

        plt.subplot(n, 2, i)
        plt.imshow(o, 
            # cmap='gray', 
            aspect='auto')
        # plt.title("Original")
        i += 1

        plt.subplot(n, 2, i)
        plt.imshow(r, 
            # cmap='gray', 
            aspect='auto')
        # plt.title("Sampled")
        i += 1

    fig.suptitle("Original vs Reconstructed Data", fontsize = TITLE_FONT_SIZE)
    fig.tight_layout()
    plt.savefig(f'{path}{signiture}_orig_vs_recon.png')
    plt.show()



def plot_samples(samples, n, path, signiture):    
    fig, axs = plt.subplots(n, 1, figsize=(6,8))
    i = 0
    for _ in range(n):
        rnd_idx = np.random.choice(len(samples))
        s = samples[rnd_idx]
        axs[i].plot(s)    
        i += 1

    fig.suptitle("Generated Samples (Scaled)", fontsize = TITLE_FONT_SIZE)
    fig.tight_layout()
    plt.savefig(f'{path}{signiture}_gen_samples_scaled.png')
    plt.show()



def plot_latent_space_timeseries(vae, n, figsize, path, signiture):
    scale = 3.0
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    grid_size = len(grid_x)

    Z2 = [ [x, y]  for x in grid_x for y in grid_y ]
    X_recon = vae.get_prior_samples_given_Z(Z2)
    X_recon = np.squeeze(X_recon)
    # print('latent space X shape:', X_recon.shape)

    
    fig, axs = plt.subplots(grid_size, grid_size, figsize=figsize)
    k = 0
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            x_recon = X_recon[k]
            k += 1            
            axs[i,j].plot(x_recon)
            axs[i,j].set_title(f'z1={np.round(xi, 2)};  z2={np.round(yi,2)}')
    
    
    fig.suptitle("Generated Samples From 2D Embedded Space", fontsize = TITLE_FONT_SIZE)
    fig.tight_layout()
    plt.savefig(f'{path}{signiture}_gen_samples_embedded.png')
    plt.show()



def plot_latent_space(vae, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 2.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    Z2 = [ [x, y]  for x in grid_x for y in grid_y ]
    X_recon = vae.get_prior_samples_given_Z(Z2)
    X_recon = np.squeeze(X_recon)
    # print(X_recon.shape)
    
    k = 0
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            x_decoded = X_recon[k]
            k += 1
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = x_decoded

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()



# Modified from TimeGAN's visualization functions
def plot_tsne(ori_data, generated_data, path, signiture):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    """  
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)  

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape  

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
        else:
            prep_data = np.concatenate((prep_data, 
                                        np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat, 
                                            np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))

    # Visualization parameter        
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    
    
    
    # Do t-SNE Analysis together       
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)

    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(prep_data_final)
        
    # Plotting
    f, ax = plt.subplots(1)
        
    plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                c = colors[:anal_sample_no], alpha = 0.5, label = "Original")
    plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                c = colors[anal_sample_no:], alpha = 0.5, label = "Synthetic")

    ax.legend()
        
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.savefig(f'{path}{signiture}_tsne.png')
    plt.show()



def plot_losses(training_history, path, signature):
    """
    Plots the training history of TimeVAE by visualizing the ELBO, Reconstruction, and KL Divergence losses.

    Parameters:
    training_history (History): The history object returned by the Keras model during training. 
                                Contains loss values and other metrics over the epochs.
    path (str): The directory path where the plot will be saved.
    signature (str): A unique identifier to append to the filename of the saved plot.

    The function performs the following tasks:
    - Creates a plot of the training history including ELBO loss, Reconstruction loss, and KL divergence loss.
    - Configures the plot with labels, log-scaled y-axis, legend, and grid.
    - Saves the plot as an image in the specified path with the given signature.
    - Displays the plot.

    Returns:
    None

    ChatGPT was used in the generation of this docstring. 
    """

    epochs = len(training_history.history["loss"])
    epochs = range(1, epochs + 1)
    plt.figure(figsize=(10, 6))

    # Plot the ELBO loss:
    plt.plot(epochs, training_history.history['loss'], 'r', label='ELBO Loss')

    # Plot the Reconstruction loss:
    plt.plot(epochs, training_history.history['reconstruction_loss'], 'g', label='Reconstruction Loss')

    # Plot the KL Divergence loss:
    plt.plot(epochs, training_history.history['kl_loss'], 'b', label='KL Divergence Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'{path}{signature}_training.png')
    plt.show()



# Custom scaler for 3d data
class MinMaxScaler_Feat_Dim():
    '''Scales history and forecast parts of time-series based on history data'''
    def __init__(self, scaling_len, input_dim, upper_bound = 3., lower_bound = -3.):         
        self.scaling_len = scaling_len
        self.min_vals_per_d = None      
        self.max_vals_per_d = None  
        self.input_dim = input_dim
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        

    def fit(self, X, y=None): 

        if self.scaling_len < 1: 
            msg = f''' Error scaling series. 
            scaling_len needs to be at least 2. Given length is {self.scaling_len}.  '''
            raise Exception(msg)

        X_f = X[ :,  : self.scaling_len , : ]
        self.min_vals_per_d = np.expand_dims(np.expand_dims(X_f.min(axis=0).min(axis=0), axis=0), axis=0)
        self.max_vals_per_d = np.expand_dims(np.expand_dims(X_f.max(axis=0).max(axis=0), axis=0), axis=0)

        self.range_per_d = self.max_vals_per_d - self.min_vals_per_d
        self.range_per_d = np.where(self.range_per_d == 0, 1e-5, self.range_per_d)

        # print(self.min_vals_per_d.shape); print(self.max_vals_per_d.shape)
              
        return self
    
    def transform(self, X, y=None): 
        assert X.shape[-1] == self.min_vals_per_d.shape[-1], "Error: Dimension of array to scale doesn't match fitted array."
         
        X = X - self.min_vals_per_d
        X = np.divide(X, self.range_per_d )        
        X = np.where( X < self.upper_bound, X, self.upper_bound)
        X = np.where( X > self.lower_bound, X, self.lower_bound)
        return X
    
    def fit_transform(self, X, y=None):
        X = X.copy()
        self.fit(X)
        return self.transform(X)
        

    def inverse_transform(self, X):
        X = X.copy()
        X = X * self.range_per_d 
        X = X + self.min_vals_per_d
        # print(X.shape)
        return X
    


class MinMaxScaler():
    """Min Max normalizer.
    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    """
    def fit_transform(self, data): 
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data


    def fit(self, data):    
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini
        return self
        

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data

    
    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data


if __name__ == '__main__':

    # data = get_daily_data()
    data = get_mnist_data()
    print(data.shape)
