import numpy as np
from sklearn.model_selection import train_test_split


"""
EXPECTED DIR STRUCTURE

/main
- custom_sines.npz
- custom_smooths.npz
- custom_data_preprocessing.py 
"""

# should be either 'sines' or 'smooths'
dataset = 'sines'
# dataset descriptor: we can have 1 wave per time-series (1w) or six waves (6w)
dataset_state = '1w'

# load data
data = np.load('custom_' + dataset + '_' + dataset_state + '.npz')
data = data['data']
print(' full data shape: {:>15}'.format(str(data.shape)))

# set random seed and split data in train and test splits
seed = np.random.seed(1)
X_train, X_test  = train_test_split(data   , test_size=0.2, random_state=seed)

# print shapes and min/max values
print('train data shape: {:>15}'.format(str(X_train.shape)))
print(' test data shape: {:>15}'.format(str(X_test.shape )))
print()
print('min and max values')
print('X_train: {: .9f}\t{: .9f}'.format(np.min(X_train), np.max(X_train)))
print(' X_test: {: .9f}\t{: .9f}'.format(np.min(X_test) , np.max(X_test)))

# save each split
np.savez('custom_' + dataset + '_' + dataset_state + '_train.npz', X_train)
np.savez('custom_' + dataset + '_' + dataset_state + '_test.npz' , X_test )
