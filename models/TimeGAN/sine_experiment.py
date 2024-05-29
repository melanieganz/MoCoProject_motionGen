"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization


def main (args):
  """Main function for timeGAN experiments.
  
  Args:
    - data_name: sine, stock, or energy
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation
  
  Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
  """

  ## Data loading
  input_file = f'../DataVault/{args[0]}_{args[1]}.npz'
  ori_data = np.load(input_file)
  try:
    ori_data = ori_data['data']
  except:
    ori_data = ori_data['arr_0']
    
  print(args[0] + ' dataset is ready.')
    
  ## Synthetic data generation by TimeGAN
  # Set newtork parameters
  parameters = dict()  
  parameters['module'] = args[2]
  parameters['hidden_dim'] = args[3]
  parameters['num_layer'] = args[4]
  parameters['iterations'] = args[5]
  parameters['batch_size'] = args[6]
      
  generated_data = timegan(ori_data, parameters)   
  print('Finish Synthetic Data Generation')
  
  ## Performance metrics   
  # Output initialization
  #metric_results = dict()
  
  # 1. Discriminative Score
  #discriminative_score = list()
  #for _ in range(args[7]):
  #  temp_disc = discriminative_score_metrics(ori_data, generated_data)
  #  discriminative_score.append(temp_disc)
      
  #metric_results['discriminative'] = np.mean(discriminative_score)
      
  # 2. Predictive score
  #predictive_score = list()
  #for tt in range(args[7]):
  #  temp_pred = predictive_score_metrics(ori_data, generated_data)
  #  predictive_score.append(temp_pred)

  # plot_path = './outputs/stage_one/plots/'
  # signiture = f"{args[2]}_{args[4]}_{args[3]}"   
      
  #metric_results['predictive'] = np.mean(predictive_score)     
          
  # 3. Visualization (t-SNE)
  # visualization(ori_data, generated_data, 'tsne', plot_path, signiture)
  
  ## Print discriminative and predictive scores
  #print(metric_results)

  return ori_data, generated_data#, metric_results


if __name__ == '__main__':  
  # Arguments:
  dataset_name      = "custom_24_sines_1w_final"
  dataset_state     = "train"
  module            = ["gru"]
  hidden_dim        = 96
  num_layer         = 3
  iteration         = 1000
  batch_size        = 32
  metric_itteration = 10

  start = time.time()

  print("\n")
  print("-------------------------------------------------------------------------------------")
  print(f"Training TimeGAN with module = {module}, nl = {num_layer}, and hd = {hidden_dim}")
  print("-------------------------------------------------------------------------------------")
  print("\n")

  # Define args:
  args = [dataset_name, dataset_state, module, hidden_dim, num_layer, iteration, batch_size, metric_itteration]

  # Calls main function  
  #ori_data, generated_data, metrics = main(args)
  ori_data, generated_data = main(args)
  
  output_dir = './outputs/stage_one/data/'
  sample_fname = f'{dataset_name}_{module}_{num_layer}_{hidden_dim}_gen.npz'

  #np.savez_compressed("../TSGBench/data/timegan/" + sample_fname, data=generated_data)
  #np.savez_compressed(output_dir + sample_fname, data=generated_data) 

  end = time.time()
  print(f"Total run time: {np.round((end - start)/60.0, 2)} minutes")
