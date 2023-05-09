
channel = 'temperature'

#%% reorganize the data - only generate noise for one feature channel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def inject_noise_single(data, labels, ratio):

    noisy_sample_size = int(len(data) * ratio)
    index = np.random.choice(len(data)-5, int(noisy_sample_size/5))

    noisy_idx = [ np.arange(id, id+5) for id in list(index)]
    noisy_idx_flatten = np.array(noisy_idx).flatten()
    # noisy_idx_flatten = np.unique(noisy_idx_flatten)

    mode = ['Noise', 'Short-term', 'Fixed']
    error_mote = []
    error_mode = []
    for index in noisy_idx:
        random_mote = np.random.choice(data.shape[1], int(data.shape[1] * 0.3), replace=True)
        fault_mode = mode[np.random.choice(3, 1, replace=True)[0]]
        for id in index:
            error_mote.append(random_mote)
            error_mode.append(fault_mode)

        if fault_mode == 'Noise': #data[index, random_mote, :, :][:, :,random_sensor]
            std = np.std(data[index,][:, random_mote], axis=0)
            data[index,][:, random_mote] = data[index,][:, random_mote] + \
                                       np.random.normal(np.zeros(data[index,][:, random_mote].shape), std*10)
        elif fault_mode == 'Short-term':
            data[index,][:, random_mote] += data[index,][:, random_mote] * 10
        elif fault_mode == 'Fixed':
            mean = np.mean(data[index,][:, random_mote], axis=0)
            data[index,][:, random_mote] = mean + 2

    labels[noisy_idx_flatten] = 1
    return noisy_idx_flatten, data, labels, error_mote, error_mode


def inject_noise_blockwise(data, labels, ratio):
    '''
    Try to inject blockwise noise
    :param data:
    :param labels:
    :param ratio:
    :return:
    '''
    noisy_sample_size = int(len(data) * ratio)
    index = np.random.choice(len(data) - 5, int(noisy_sample_size / 5), replace=True)

    noisy_idx = [np.arange(id, id + 5) for id in list(index)]
    noisy_idx_flatten = np.array(noisy_idx).flatten()
    # noisy_idx_flatten = np.unique(noisy_idx_flatten)

    mode = ['Noise', 'Short-term', 'Fixed']
    error_mote = []
    error_mode = []
    # random_mote = np.random.choice(data.shape[1], int(data.shape[1] * 0.3), replace=True)
    for ii, index in enumerate(noisy_idx):
        if ii% 100 == 0:
            random_mote = np.random.choice(data.shape[1], int(data.shape[1] * 0.3), replace=True)

        fault_mode = mode[np.random.choice(3, 1, replace=True)[0]]
        for id in index:
            error_mote.append(random_mote)
            error_mode.append(fault_mode)

        if fault_mode == 'Noise':  # data[index, random_mote, :, :][:, :,random_sensor]
            std = np.std(data[index,][:, random_mote], axis=0)
            data[index,][:, random_mote] = data[index,][:, random_mote] + \
                                           np.random.normal(np.zeros(data[index,][:, random_mote].shape), std * 10)
        elif fault_mode == 'Short-term':
            data[index,][:, random_mote] += data[index,][:, random_mote] * 10
        elif fault_mode == 'Fixed':
            mean = np.mean(data[index,][:, random_mote], axis=0)
            data[index,][:, random_mote] = mean + 2

    labels[noisy_idx_flatten] = 1
    return noisy_idx_flatten, data, labels, error_mote, error_mode

#%%

data_path = r'C:\Users\Thaibite Zeng\OneDrive - Virginia Tech\Virginia Tech\Courses\2023 Spring\CS6804\Project'
save_path = r'C:\Users\Thaibite Zeng\VT_research\GraphNN\data\{}'.format(channel)
df = pd.read_csv(data_path + '/results/processed_data.csv')
moteids_to_remove = [5, 15, 18, 28, 54, 55, 56, 58]
filtered_df = df[~df['moteid'].isin(moteids_to_remove)]

grouped_by_moteid = filtered_df.groupby('moteid')
feature_cols = ['temperature', 'humidity', 'voltage', 'light']

temp_all = {}
temp_data = filtered_df[['time_formatted', channel, 'moteid']].groupby('moteid')

min_length = []
for moteid, moteid_data in temp_data:
    moteid_data = moteid_data.sort_values(by='time_formatted', ascending=True)
    min_length.append(len(moteid_data))
    temp_all[moteid] = moteid_data

min_len = min(min_length)
for moteid, moteid_data in temp_data:
    moteid_data = moteid_data.sort_values(by='time_formatted', ascending=True)
    temp_all[moteid] = temp_all[moteid][:min_len]

all_moteid = list(temp_all.keys())
all_values = []
for value in temp_all.values():
    all_values.append(value[channel].values)
all_values = np.array(all_values).reshape(len(all_moteid),-1).T
labels = np.zeros(all_values.shape[0])

training_ratio = 0.6
training_size =  11900 #int(all_values.shape[0]*training_ratio)
training_data = all_values[:training_size]


noise_test = all_values[training_size: training_size + 8000]
test_labels = labels[training_size: training_size + 8000]
ratio = 0.33
noisy_idx_flatten, noise_test, test_labels , error_mote, error_mode = inject_noise_blockwise(noise_test, test_labels, ratio)

np.save(save_path + '/testing_error_mote_{}'.format(channel), np.array(error_mote))
np.save(save_path + '/testing_error_mode_'.format(channel), np.array(error_mode))
np.save(save_path + '/testing_error_index', np.array(noisy_idx_flatten))
#%%  Normalization
from sklearn.preprocessing import MinMaxScaler

# max min(0-1)
def norm(train, test):

    normalizer = MinMaxScaler(feature_range=(0, 1)).fit(train) # scale training data to [0,1] range
    train_ret = normalizer.transform(train)
    test_ret = normalizer.transform(test)

    return train_ret, test_ret

strings = ["M-" + str(int(num)) for num in all_moteid]

x_train, x_test = norm(training_data, noise_test)
df_train = pd.DataFrame(x_train, columns=strings)
df_train['timestamp'] = range(len(df_train))
first_column = df_train.pop('timestamp')
df_train.insert(0, 'timestamp', first_column)

df_train.to_csv(save_path + '/{}_train.csv'.format(channel), index=False)

column_names = list(df_train.columns)[1:]
with open(save_path + '/list.txt', 'w') as file:
    for name in column_names:
        file.write(name + '\n')

test_data = np.concatenate((x_test, test_labels.reshape(-1, 1)), axis=1)
df_test = pd.DataFrame(test_data, columns=strings + ['attack'])
df_test['timestamp'] = range(len(df_test))
first_column = df_test.pop('timestamp')
df_test.insert(0, 'timestamp', first_column)
df_test.to_csv(save_path + '/{}_test.csv'.format(channel), index=False)


#%% visualize
mote_path = r'C:\Users\Thaibite Zeng\VT_research\GraphNN\data\{}'.format(channel)
abnormal_index = np.load(mote_path + '/testing_error_index.npy')
mote_mtrx = np.zeros((x_test.shape[0], len(strings)))
error_mote_humid = np.load(mote_path + '/testing_error_mote_{}.npy'.format(channel))

for ii, index in enumerate(abnormal_index):
    if index not in moteids_to_remove:
        index = index[index < mote_mtrx.shape[0]]
        # print(len(index))
        if len(index) > 0:
            mote_mtrx[index, error_mote_humid[ii]] = 1

plt.imshow(mote_mtrx, aspect='auto', cmap='viridis')
plt.show()

#%% generate some abnormal case in training for the LSTM case study
# train test splitting - validation generation
from sklearn.model_selection import train_test_split

save_path = r'C:\Users\Thaibite Zeng\VT_research\GraphNN\data\{}'.format(channel)
noise_train = all_values[: training_size]
train_labels = labels[: training_size]
ratio = 0.05
validation_data = all_values[training_size + 8000: training_size + 10000]
validation_labels = labels[training_size + 8000: training_size + 10000]

noise_train, validation_data = norm(noise_train, validation_data)

noisy_idx_flatten, noise_train, train_labels , error_mote, error_mode = inject_noise_blockwise(noise_train, train_labels, ratio)
noisy_idx_flatten, noise_valid, valid_labels , error_mote, error_mode = inject_noise_blockwise(validation_data, validation_labels, ratio)


normal_train = np.where(train_labels == 0)[0]
normal_train = noise_train[normal_train]
normal_label = np.zeros(len(normal_train))
# Saving training data (only normal instances)
np.save(save_path + '/numpy/X_train.npy', normal_train)
np.save(save_path + '/numpy/y_train.npy', normal_label)

# Saving training data (normal instances + anomalous)
np.save(save_path + '/numpy/X_train_p.npy', noise_train)
np.save(save_path + '/numpy/y_train_p.npy', train_labels)

# Saving validation data (only normal instances to control model training)

normal_valid = np.where(valid_labels == 0)[0]
normal_valid = noise_valid[normal_valid]
normal_label = np.zeros(len(normal_valid))

# Saving validation data (only normal instances to control model training)
np.save(save_path + '/numpy/X_val.npy', normal_valid)
np.save(save_path + '/numpy/y_val.npy',  normal_label)

np.save(save_path + '/numpy/X_val_p.npy', noise_valid)
np.save(save_path + '/numpy/y_val_p.npy', valid_labels)

np.save(save_path + '/numpy/X_val_p_full.npy', noise_valid)
np.save(save_path + '/numpy/y_val_p_full.npy', valid_labels)

# Saving test data (normal + anomalous instances)
np.save(save_path + '/numpy/X_test.npy', x_test)
np.save(save_path + '/numpy/y_test.npy', test_labels)