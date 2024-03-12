import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import MinMaxScaler
from sklearn.model_selection import train_test_split


"""
EXPECTED DIR STRUCTURE

/main
- /descriptions
  - participants.tsv
- /fmri
  - /sub-...
    - /ses-baselineYear1Arm1
      - /func
        - sub-..._motion.tsv
        - sub-..._motion.tsv
  - /sub-...
  - /sub-...
- fmri_data_preprocessing.py 
- utils.py
"""

def load_desc():
    # load subject descriptions
    desc_path = 'descriptions/participants.tsv'
    desc = pd.read_csv(desc_path, delimiter='\t')
    return desc

def load_subj():
    # load subject names
    subj = np.array(os.listdir('fmri'))
    return subj

def filter_subj_desc(subj, desc):
    # preprocess subjects
    subj_desc = desc[desc['participant_id'].isin(subj)]
    subj_desc = subj_desc[subj_desc['session_id'].isin(['ses-baselineYear1Arm1'])]
    subj_desc = subj_desc[subj_desc['participant_id'] != 'sub-NDARINV66PM75JX'] 
    subj_desc = subj_desc[subj_desc['age'] != 888] 
    subj_desc['age'] = subj_desc['age'] // 12
    return subj_desc

def load_bar(counter, count_to, count_every):
    # primitive load bar
    if (counter + 1) % count_every == 0:
        print(f'{counter + 1}/{count_to}', end='\r', flush=True)

def create_data_arrays(subj_desc):
    # create data array, histogram array and information array
    data_array = []
    hist_array = []
    info_array = []

    # iterate over all subjects
    for i,subj in enumerate(subj_desc['participant_id']):
        # visualize progress
        load_bar(i, len(subj_desc), 100)

        # only consider baseline year 1 runs
        runs_dir = 'fmri/' + subj + '/ses-baselineYear1Arm1/func'  
        for file in os.listdir(runs_dir):
            # only consider first run each participant had
            if file.endswith('rest_run-1_desc-includingFD_motion.tsv'):
                # grab file path
                file_path = runs_dir + '/' + file

                # read displacement parameters of each csv file
                data = pd.read_csv(file_path, delimiter=r'\s+', usecols=['trans_x_mm', 'trans_y_mm', 'trans_z_mm', 'rot_x_degrees', 'rot_y_degrees', 'rot_z_degrees'])
                
                # grab number of frames
                frames = len(data)
                hist_array.append(frames)
                
                # skip runs with less than 380 frames, truncate and store information of runs with more than 380 frames
                if frames < 380:
                    continue
                data_array.append(data[:380])
                row = subj_desc[subj_desc['participant_id'] == subj]
                age = row['age'].iloc[0]
                sex = row['sex'].iloc[0]
                info_array.append([subj, age, sex])
    
    # convert to numpy arrays
    data_array = np.array(data_array)
    hist_array = np.array(hist_array)
    info_array = np.array(info_array)
    return data_array, hist_array, info_array

def plot_histogram(hist_array):
    # create histogram and plot it
    n, bins, _ = plt.hist(hist_array, bins=100, range=(350, 400), log=True, edgecolor='black')
    threshold = 0
    bar_centers = (bins[:-1] + bins[1:]) / 2
    selected_bar_centers = bar_centers[n > threshold]
    selected_bar_centers = [int(x) for x in selected_bar_centers]
    plt.xticks(selected_bar_centers, rotation='vertical', fontsize=8)
    plt.title('run 1 of all participants')
    plt.xlabel('number of frames')
    plt.ylabel('number of participants (log scale)')
    plt.show()


# load data and create data arrays
desc = load_desc()
subj = load_subj()
subj_desc = filter_subj_desc(subj, desc)
data_array, hist_array, info_array = create_data_arrays(subj_desc)
print('desc shape:       {:>15}'.format(str(desc.shape)))
print('subj shape:       {:>15}'.format(str(subj.shape)))
print('subj_desc shape:  {:>15}'.format(str(subj_desc.shape)))
print('data_array shape: {:>15}'.format(str(data_array.shape)))
print('info_array shape: {:>15}'.format(str(info_array.shape)))
print('hist_array shape: {:>15}'.format(str(hist_array.shape)))
print()

# plot histogram
plot_histogram(hist_array)
print('(10 or more occurences of same frame number)')
print('frames', '\t', '# participants')
uniques = np.unique(hist_array)
for unique in uniques:
    count = len(np.where(hist_array == unique)[0])
    if count >= 10:
        print(unique, '\t', count)
print('number of participants with less than 380 frames in run 1: ', len(np.where(np.array(hist_array) < 380)[0]))
print('number of participants with more than 379 frames in run 1:' , len(np.where(np.array(hist_array) > 379)[0]))
print()


# set random seed and split data and info in train, validation and test
seed = np.random.seed(1)
X_train, X_test , info_train, info_test  = train_test_split(data_array, info_array, test_size=0.2, random_state=seed)
X_train, X_valid, info_train, info_valid = train_test_split(X_train,    info_train, test_size=0.2, random_state=seed)
print('train data shape: {:>15}'.format(str(X_train.shape)))
print('valid data shape: {:>15}'.format(str(X_valid.shape)))
print(' test data shape: {:>15}'.format(str(X_test.shape )))
print('train info shape: {:>15}'.format(str(info_train.shape)))
print('valid info shape: {:>15}'.format(str(info_valid.shape)))
print(' test info shape: {:>15}'.format(str(info_test.shape )))
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
print('min and max after scaling')
print('X_train: {: .9f}\t{: .9f}'.format(np.min(X_train), np.max(X_train)))
print('X_valid: {: .9f}\t{: .9f}'.format(np.min(X_valid), np.max(X_valid)))
print('X_test:  {: .9f}\t{: .9f}'.format(np.min(X_test) , np.max(X_test)))

# save each split
np.savez('fmri_train.npz', X_train)
np.savez('fmri_valid.npz', X_valid)
np.savez('frmi_test.npz' , X_test )