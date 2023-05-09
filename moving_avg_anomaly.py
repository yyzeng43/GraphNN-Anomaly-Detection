# anomaly detection by moving average method

from itertools import count
import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
import pandas as pd
import collections
from random import randint
import matplotlib

def mov_average(data, window_size):

    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')


def find_anomalies(y, window_size, sigma=1.0):
    avg = mov_average(y, window_size).tolist()
    residual = y - avg
    std = np.std(residual)
    return {'standard_deviation': round(std, 3),
            'anomalies_dict': collections.OrderedDict([(index, y_i) for index, y_i, avg_i in zip(count(), y, avg)
              if (y_i > avg_i + (sigma*std)) | (y_i < avg_i - (sigma*std))])}


def plot_results(x, y, window_size, sigma_value=1,
                 text_xlabel="X Axis", text_ylabel="Y Axis", applying_rolling_std=False, abnormal_index = None):
    plt.figure(figsize=(15, 8))
    font = {'family': 'normal',
            'size': 20}

    matplotlib.rc('font', **font)

    plt.plot(x, y, "k.")
    y_av = mov_average(y, window_size)
    plt.plot(x, y_av, color='green')
    # plt.xlim(0, 40000)
    plt.xlabel(text_xlabel)
    plt.ylabel(text_ylabel)
    events = {}
    events = find_anomalies(y, window_size=window_size, sigma=sigma_value)

    x_anom = np.fromiter(events['anomalies_dict'].keys(), dtype=int, count=len(events['anomalies_dict']))
    y_anom = np.fromiter(events['anomalies_dict'].values(), dtype=float, count=len(events['anomalies_dict']))
    plt.plot(x_anom, y_anom, "r*", label = 'Detected Abnormal')
    print(x_anom)
    plt.grid(True)


    if abnormal_index is not None:
        plt.plot(abnormal_index, y[abnormal_index], "b^", label = 'Actual Abnormal')

    plt.legend()
    plt.show()
    return x_anom



#%% check the obnormal
from sklearn.metrics import f1_score, accuracy_score, recall_score

channel = 'temperature'
data_path =  r'C:\Users\Thaibite Zeng\VT_research\GraphNN\data\{}'.format(channel)

temp_df_train = pd.read_csv(data_path + '/{}_train.csv'.format(channel))
temp_df_test = pd.read_csv(data_path + '/{}_test.csv'.format(channel))
all_abnormal_index = np.where(temp_df_test['attack'].values == 1)[0]


mote_path = r'C:\Users\Thaibite Zeng\VT_research\GraphNN\data\{}'.format(channel)
abnormal_index = np.load(mote_path + '/testing_error_index.npy')
error_mote_humid = np.load(mote_path + '/testing_error_mote_{}.npy'.format(channel))

mote_id = 16
mote_label = np.zeros(temp_df_test.shape[0])
for ii, index in enumerate(abnormal_index):
    if mote_id in error_mote_humid[ii]:
        mote_label[index] = 1

mote_label_id = np.where(mote_label==1)[0]
x, y = np.arange(1, temp_df_test.shape[0]+1), temp_df_test['M-{}'.format(mote_id)]
_ = plot_results(x, y=y, window_size=50, text_xlabel="Time Stamp", sigma_value=3,text_ylabel=channel,
                           abnormal_index=mote_label_id)

abnormal_events = find_anomalies(y, window_size=50, sigma=3)
abnormal_ids = list(abnormal_events['anomalies_dict'].keys())
y_pred = np.zeros(temp_df_test.shape[0])
y_pred[abnormal_ids] = 1

f1, acc, recall = f1_score(y_pred, mote_label), accuracy_score(y_pred, mote_label), recall_score(y_pred, mote_label)

print(f1, acc, recall)

save_path = r'C:\Users\Thaibite Zeng\VT_research\GraphNN\results\{}'.format(channel)
np.save(save_path + '/y_pred_MA_M1{}'.format(mote_id),y_pred)

