import numpy as np
from utils import MinMaxScaler
from sklearn.model_selection import train_test_split


"""
EXPECTED DIR STRUCTURE

/main
- custom_sines.npz
- custom_data_preprocessing.py 
- utils.py
"""

# should be either 'sines' or 'smooths'
dataset = 'sines'

# load data
data = np.load('custom_' + dataset + '.npz')
data = data['data']
print(' full data shape: {:>15}'.format(str(data.shape)))

# set random seed and split data and info in train, validation and test
seed = np.random.seed(1)
X_train, X_test  = train_test_split(data   , test_size=0.2, random_state=seed)
X_train, X_valid = train_test_split(X_train, test_size=0.2, random_state=seed)

# print shapes and min/max values before scaling
print('train data shape: {:>15}'.format(str(X_train.shape)))
print('valid data shape: {:>15}'.format(str(X_valid.shape)))
print(' test data shape: {:>15}'.format(str(X_test.shape )))
print()
print('min and max before scaling')
print('X_train: {: .9f}\t{: .9f}'.format(np.min(X_train), np.max(X_train)))
print('X_valid: {: .9f}\t{: .9f}'.format(np.min(X_valid), np.max(X_valid)))
print('X_test:  {: .9f}\t{: .9f}'.format(np.min(X_test) , np.max(X_test)))
print()

# scale data
scaler  = MinMaxScaler()        
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test  = scaler.transform(X_test)

# print min/max values after scaling
print('min and max after scaling')
print('X_train: {: .9f}\t{: .9f}'.format(np.min(X_train), np.max(X_train)))
print('X_valid: {: .9f}\t{: .9f}'.format(np.min(X_valid), np.max(X_valid)))
print('X_test:  {: .9f}\t{: .9f}'.format(np.min(X_test) , np.max(X_test)))

# save each split
np.savez('custom_' + dataset + '_train.npz', X_train)
np.savez('custom_' + dataset + '_valid.npz', X_valid)
np.savez('custom_' + dataset + '_test.npz' , X_test )