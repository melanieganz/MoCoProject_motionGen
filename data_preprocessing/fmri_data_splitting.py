import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator
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
- fmri_data_splitting.py 
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
    subj_desc = subj_desc[subj_desc['age'].isin([9, 10])]
    subj_desc = subj_desc[subj_desc['sex'].isin([1,  2])]
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

    # convert degrees to millimeters
    data_array[:, :, 3:] = 50 * np.deg2rad(data_array[:, :, 3:])
    print("shape:", data_array.shape)

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

def remove_outliers(data_array, info_array, percentile):
    rows, _, dims = data_array.shape

    # find maximum displacement values
    max_disp = []
    for i in range(rows):
        temp = []
        for d in range(dims):
            if abs(min(data_array[i, :, d])) > abs(max(data_array[i, :, d])):
                temp.append(min(data_array[i, :, d]))
            else:
                temp.append(max(data_array[i, :, d]))
        max_disp.append(temp)
    max_disp = np.array(max_disp)

    # compute boxplot with whiskers
    bp = plt.boxplot([max_disp[:, d] for d in range(dims)], whis=percentile)
    plt.close()

    # extract whiskers
    whiskers = [item.get_ydata()[1] for item in bp['whiskers']] 
    
    # remove outliers
    new_data = []
    new_info = []
    for i in range(rows):
        temp_data = []
        temp_info = []
        keep = True
        for d in range(dims):
            # extract lower and upper whisker per dimension
            low_whisk, high_whisk = whiskers[d*2:(d+1)*2]
            if not (any(data_array[i, :, d] < low_whisk) or any(data_array[i, :, d] > high_whisk)):
                # only keep data that is between the whiskers
                temp_data.append(data_array[i, :, d])
                temp_info.append(info_array[i])
            else:
                keep = False
        if keep:
            new_data.append(temp_data)
            new_info.append(temp_info)

    # convert to numpy arrays and fix shape
    new_data = np.array(new_data).transpose(0, 2, 1)
    new_info = np.array(new_info).transpose(0, 2, 1)

    return new_data, new_info

def violinplot(data_array, labels):
    rows, _, dims = data_array.shape

    # Find maximum displacement values
    max_disp = []
    for i in range(rows):
        temp = []
        for d in range(dims):
            if abs(min(data_array[i, :, d])) > abs(max(data_array[i, :, d])):
                temp.append(min(data_array[i, :, d]))
            else:
                temp.append(max(data_array[i, :, d]))
        max_disp.append(temp)
    max_disp = np.array(max_disp)

    # Compute violinplot
    vp = plt.violinplot([max_disp[:, d] for d in range(dims)],
                        vert=True,
                        widths=0.5,
                        showmedians=True,
                        showmeans=True,
                        showextrema=True)
    
    # Set colors for violinplot
    for patch in vp['bodies']:
        patch.set_edgecolor('black')
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp[partname].set_edgecolor('black')
        vp[partname].set_linewidth(1)
    vp['cmedians'].set_edgecolor('red')
    vp['cmedians'].set_linewidth(1)
    
    plt.ylabel('Millimeters [mm]', fontsize=11, weight='bold')
    plt.xticks(np.arange(1, dims + 1), labels, weight='bold')
    plt.yticks(weight='bold')
    plt.grid()
    plt.tight_layout()
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

# needed for violin plots and printouts
categories = ['trans-X', 'trans-Y', 'trans-Z', 'rot-X', 'rot-Y', 'rot-Z']

# violin plot before outlier removal
violinplot(data_array[:, :, :], labels=categories)
for d in range(6):
    print(categories[d] + ': {:3.2f}\t{:3.2f}'.format(np.min(data_array[:, :, d]), np.max(data_array[:, :, d])))

# remove outliers
data_array, info_array = remove_outliers(data_array, info_array, percentile=(2.5, 97.5))
rows, cols, dims = data_array.shape

# violin plot after outlier removal
violinplot(data_array[:, :, :], labels=categories)
for d in range(6):
    print(categories[d] + ': {:3.2f}\t{:3.2f}'.format(np.min(data_array[:, :, d]), np.max(data_array[:, :, d])))

# plot histogram
plot_histogram(hist_array)
print('(10 or more occurences of same frame number)')
print('frames', '\t', '# participants')
uniques = np.unique(hist_array)
for unique in uniques:
    count = len(np.where(hist_array == unique)[0])
    if count >= 1:
        print(unique, '\t', count)
print('number of participants with less than 380 frames in run 1: ', len(np.where(np.array(hist_array) < 380)[0]))
print('number of participants with more than 379 frames in run 1:' , len(np.where(np.array(hist_array) > 379)[0]))
print()

# set random seed and split data and info in train and test splits
seed = np.random.seed(1)
X_train, X_test , info_train, info_test  = train_test_split(data_array, info_array, test_size=0.2, random_state=seed)
X_train, X_valid , info_train, info_valid  = train_test_split(X_train, info_train, test_size=0.1, random_state=seed)
print('train data shape: {:>15}'.format(str(X_train.shape)))
print('valid data shape: {:>15}'.format(str(X_valid.shape)))
print(' test data shape: {:>15}'.format(str(X_test.shape )))
print('train info shape: {:>15}'.format(str(info_train.shape)))
print('valid info shape: {:>15}'.format(str(info_valid.shape)))
print(' test info shape: {:>15}'.format(str(info_test.shape )))
print()
print('min and max values')
print('X_train: {: .9f}\t{: .9f}'.format(np.min(X_train), np.max(X_train)))
print('X_valid: {: .9f}\t{: .9f}'.format(np.min(X_valid), np.max(X_valid)))
print('X_test:  {: .9f}\t{: .9f}'.format(np.min(X_test) , np.max(X_test)))
print()

# maximum and minimum values per curve per dataset
print('X_TRAIN')
for d in range(dims):
    print(categories[d] + ': {:3.9f}\t{:3.9f}'.format(np.min(X_train[:, :, d]), np.max(X_train[:, :, d])))
print()
print('X_VALID')
for d in range(dims):
    print(categories[d] + ': {:3.9f}\t{:3.9f}'.format(np.min(X_valid[:, :, d]), np.max(X_valid[:, :, d])))
print()
print('X_TEST')
for d in range(dims):
    print(categories[d] + ': {:3.9f}\t{:3.9f}'.format(np.min(X_test[:, :, d]), np.max(X_test[:, :, d])))

# set to True to save each split
if False:
    np.savez('fmri_train_new.npz', X_train)
    np.savez('info_train_new.npz', info_train)
    np.savez('fmri_valid_new.npz', X_valid)
    np.savez('info_valid_new.npz', info_valid)
    np.savez('fmri_test_new.npz' , X_test)
    np.savez('info_test_new.npz' , info_test)
