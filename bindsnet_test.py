#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/2 13:44
# @Author  : Kang
# @Site    : 
# @File    : bindsnet_test.py
# @Software: PyCharm
import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
from bindsnet_data_lib import get_model_data, train_test_split, bindsnet_load_data, get_pct_quantile
from bindsnet_train_lib import Trainer, plot_result, spike_accuracy, reversion_strategy

print_interval = 10; plot = False
# The parameters to tune.
window = 1
n_neurons = 64; test_percent = 0.001; n_total = None
time = 6; dt = 1.0
method = "poisson"
single_in = False
hidden = 'H' if single_in else 'H1'

# Intra-day high frequency data
# For data pre-processing
new_mean = 45
new_std = 20.0

with open('date_map.json', 'r') as fd:
    date_map = json.load(fd)
    fd.close()

commodity = 'CL'  # The commodity to be tested.
commodity_train = 'GC'  # The commodity that was used to train the model.
directory = './data/201705/Trades/'
file_list = os.listdir(directory)

data_list = [x for x in file_list if x[-9:-7] == commodity]
print(data_list)


accuracy_record = {}

for file_name in data_list:
    print('Testing data file: ' + file_name)
    date_str = file_name[13:21]
    data_path = directory + file_name

    raw_model_data, dominant = get_model_data(data_path=data_path, data_type='price', length=n_total)
    quantile = get_pct_quantile(dominant, q=0.5)
    print('Baseline mean return for spike: %.8f' % quantile)

    raw_model_data = torch.cat((torch.Tensor([0]), raw_model_data[1:] - raw_model_data[:-1]))

    test_data, _ = train_test_split(raw_model_data, test_percent=test_percent, n_train=None, n_test=None,
                                             normal=True, new_mean=new_mean, new_std=new_std)

    test_data = test_data.repeat(window, 1).transpose(0, 1)

    if not single_in:
        test_data1, _ = train_test_split(-1 * raw_model_data, test_percent=test_percent, n_train=None, n_test=None,
                                                   normal=True, new_mean=new_mean, new_std=new_std)

        test_data1 = test_data1.repeat(window, 1).transpose(0, 1)
        test_data = torch.cat([test_data, test_data1], dim=1)

    print(test_data.shape)
    print(test_data[test_data < 0])
    test_data[test_data < 0] = 0

    out_layer = 'Y'

    # model_file = './' + commodity_train + '_models/' + commodity_train + '_' + date_map[date_str] + '_model.pkl'
    model_file = './' + commodity_train + '_models/' + commodity_train + '_' + date_str + '_model.pkl'  # Used for transfer learning.
    print('Using model: ' + model_file)
    with open(model_file, 'rb') as f:
        ex = pickle.load(f)
        f.close()
    # Note that here we use the trained model from other dataset to test on this dataset. So we use the former train_data as test_data.
    n_test = len(test_data)
    test_data_loader = bindsnet_load_data(dataset=test_data, time=time, dt=dt, method=method)
    test_record = ex.testing(test_data_loader, n_test, out_layer=out_layer, plot=plot)

    plot_result(data=dominant.iloc[:n_test], spike_record=test_record, display_time=10, file_name='./images/test.png',
                plot_every=True)
    acc = spike_accuracy(data=dominant.iloc[:n_test], spike_record=test_record, base_ret=quantile, window=3)
    print('Test accuracy: ', acc)
    net_value = reversion_strategy(data=dominant.iloc[:n_test], spike_record=test_record, hold_window=3, window=3)
    print('Test net_value: ', net_value)
    ex.net_model.network.reset_()
    accuracy_record[date_str] = acc
    # net_value.to_csv('./strats/' + commodity + '_' + date_str + '_test.csv', index=False, header=True)
    net_value.to_csv('./strats/' + commodity + '_' + date_str + '_transfer.csv', index=False, header=True)


accuracy_df = np.transpose(pd.DataFrame(accuracy_record))
print(accuracy_df)

# accuracy_df.to_csv('./accuracy/' + commodity + '_test.csv', index=True)
accuracy_df.to_csv('./accuracy/' + commodity + '_transfer.csv', index=True)
