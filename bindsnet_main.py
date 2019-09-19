# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:02:14 2019
@author: kang
"""
import os
import pickle
import torch
import numpy as np
import pandas as pd
from bindsnet_data_lib import get_model_data, train_test_split, bindsnet_load_data, get_pct_quantile
from bindsnet_train_lib import Trainer, plot_result, spike_accuracy, reversion_strategy, momentum_strategy
from bindsnet_models import MultiLayerModel, DoubleInputModel

gpu = False
# seed = 0
# if gpu:
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
#     torch.cuda.manual_seed_all(seed)
# else:
#     torch.manual_seed(seed)
print_interval = 10; plot = False
# The parameters to tune.
window = 1
n_neurons = 64; test_percent = 0.001; n_total = None
epoch = 12
time = 10; dt = 1.0
method = "poisson"
single_in = False
hidden = 'H' if single_in else 'H1'

# Intra-day high frequency data
# For data pre-processing
new_mean = 60
new_std = 20.0
# data_path = './data/201705/Trades/CME.20170524-20170525.F.Trades.382.CL.csv.gz'

commodity = 'GC'
directory = './data/201705/Trades/'
file_list = os.listdir(directory)

data_list = [x for x in file_list if x[-9:-7] == commodity]
# data_list = ['CME.20170524-20170525.F.Trades.382.CL.csv.gz']
print(data_list)

for file_name in data_list:
    date_str = file_name[13:21]
    data_path = directory + file_name

    raw_model_data, dominant = get_model_data(data_path=data_path, data_type='price', length=n_total)
    quantile = get_pct_quantile(dominant, q=0.5)
    print('Baseline mean return for spike: %.8f' % quantile)

    raw_model_data = torch.cat((torch.Tensor([0]), raw_model_data[1:] - raw_model_data[:-1]))

    train_data, test_data = train_test_split(raw_model_data, test_percent=test_percent, n_train=None, n_test=None,
                                             normal=True, new_mean=new_mean, new_std=new_std)

    train_data = train_data.repeat(window, 1).transpose(0, 1)
    test_data = test_data.repeat(window, 1).transpose(0, 1)

    if not single_in:
        train_data1, test_data1 = train_test_split(-1 * raw_model_data, test_percent=test_percent, n_train=None, n_test=None,
                                                   normal=True, new_mean=new_mean, new_std=new_std)

        train_data1 = train_data1.repeat(window, 1).transpose(0, 1)
        test_data1 = test_data1.repeat(window, 1).transpose(0, 1)
        train_data = torch.cat([train_data, train_data1], dim=1)
        test_data = torch.cat([test_data, test_data1], dim=1)

    print(train_data.shape)
    print(train_data[train_data < 0])
    train_data[train_data < 0] = 0
    test_data[test_data < 0] = 0

    n_train = len(train_data)
    # Build network model.
    out_layer = 'Y'

    # net = MultiLayerModel(n_inpt=window, n_neurons=n_neurons, n_output=1, dt=dt, initial_w=10, wmin=None, wmax=None, nu=(1, 1), norm=512)
    # print('MultiLayerModel, hidden size = %d!' % n_neurons)
    net = DoubleInputModel(n_neurons=n_neurons, n_output=1, dt=dt, initial_w=10, wmin=None, wmax=None, nu=(1, 1), norm=512)
    print('DoubleInputModel, hidden size = %d and %d!' % (n_neurons, n_neurons))

    ex = Trainer(net, time, single_in=single_in, print_interval=print_interval)
    before = ex.net_model.network.connections[(hidden, 'Y')].w.clone().numpy()
    print(before)
    accuracy_record = {}
    net_value_record = {}
    for i in range(epoch):
        print('Training epoch number: %d' % (i + 1))
        # Lazily encode data as spike trains.
        train_data_loader = bindsnet_load_data(dataset=train_data, time=time, dt=dt, method=method)
        train_record = ex.training(train_data_loader, n_train, out_layer=out_layer, plot=plot, normalize_weight=True)
        middle = ex.net_model.network.connections[(hidden, 'Y')].w.clone().numpy()
        print(middle)
        plot_result(data=dominant.iloc[:n_train], spike_record=train_record, display_time=10, file_name='./images/epoch'+str(i)+'.png',
                    plot_every=False, epoch=i) # Change plot_every
        acc = spike_accuracy(data=dominant.iloc[:n_train], spike_record=train_record, base_ret=quantile, window=3)
        print('Epoch %d, accuracy: ' % i, acc)
        net_value = momentum_strategy(data=dominant.iloc[:n_train], spike_record=train_record, hold_window=3, window=3)
        print('Epoch %d, net_value: ' % i, net_value)
        accuracy_record[i] = acc
        net_value_record[i] = net_value
        ex.net_model.network.reset_()
    # save the model
    with open('./' + commodity + '_models/' + commodity + '_' + date_str + '_model.pkl', 'wb') as f_out:
        pickle.dump(ex, f_out)
        f_out.close()

    """
    # ############# The following is for testing ####################################################
    print('Training finished, begin testing!!!!!!')
    n_test = len(test_data)
    test_data_loader = bindsnet_load_data(dataset=test_data, time=time, dt=dt, method=method)
    test_record = ex.testing(test_data_loader, n_test, out_layer=out_layer, plot=plot)
    plot_result(data=dominant.iloc[n_train: n_train+n_test], spike_record=test_record, display_time=10, file_name='./images/test.png',
                plot_every=True)
    acc = spike_accuracy(data=dominant.iloc[n_train: n_train+n_test], spike_record=test_record, base_ret=quantile, window=3)
    print('Testing accuracy: ', acc)
    ex.net_model.network.reset_()
    """

    after = ex.net_model.network.connections[(hidden, 'Y')].w.clone().numpy()
    print(after)
    print(before == middle)
    print(middle == after)
    accuracy_df = np.transpose(pd.DataFrame(accuracy_record))
    print(accuracy_df)
    for key in net_value_record:
        print(key, net_value_record[key])

    accuracy_df.to_csv('./accuracy/' + commodity + '_' + date_str + '.csv', index=False)
    with open('./strats/' + commodity + '_' + date_str + '.pkl', 'wb') as f:
        pickle.dump(net_value_record, f)
        f.close()
