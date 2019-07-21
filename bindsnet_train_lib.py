#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/21 16:03
# @Author  : Kang
# @Site    : 
# @File    : bindsnet_train_lib.py
# @Software: PyCharm
import torch
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
from bindsnet.analysis.plotting import plot_spikes


def assembly_plot(monitors, *args):
    """
    To plot in the training and testing process.
    :param monitors: The monitors of the network.
    :param args: Other arguments about the plots.
    :return: args used in the next plots.
    """
    if len(args) != 0:
        spike_ims, spike_axes = args
    else:
        spike_ims = spike_axes = None
    spike_ims, spike_axes = plot_spikes({layer: monitors[layer].get('s') for layer in monitors},
                                        ims=spike_ims, axes=spike_axes)
    plt.pause(1e-8)
    return spike_ims, spike_axes


def add_spike_column(data, spike_record):
    """
    This function is used to copy the data and add a "spike" column to the data, according to spikes in "spike_record".
    :param data:pd.DataFrame, contains EntryPrize(vwap), and has integer index.
    :param spike_record: Torch.Tensor, shape: (time_stamps, n_neurons, simulation_time)
    :return: the dataset with "spike" column, pd.DataFrame.
    """
    assert len(data) == spike_record.shape[0], "The data and spikes should have the same length!"
    dataset = data.copy()
    spike_series = spike_record.view(len(dataset), -1).sum(dim=-1).clamp(0, 1)
    dataset['spike'] = spike_series
    return dataset


def reversion_strategy(data, spike_record, hold_window=3, initial_cap=1.0, window=3):
    """
    Use the reversion strategy, regard the spike as reversion point.
    :param data: pd.DataFrame, contains EntryPrize(vwap), and has integer index.
    :param spike_record: Torch.Tensor, shape: (time_stamps, n_neurons, simulation_time)
    :param hold_window: The length of each holding period.
    :param initial_cap: Initial capital of the strategy, default 1; each trade invest "initial_cap" money; no leverage.
    :param window: The window length to determine long or short for the strategy, default 3.
    :return: The net value series of the strategy, pd.Series.
    """
    dataset = add_spike_column(data=data, spike_record=spike_record)
    tmp = dataset.loc[dataset.spike == 1]
    net_value = [initial_cap]
    ret_series = []
    for i, ind in enumerate(tmp.index):
        if ind - window < dataset.index[0] or ind + hold_window > dataset.index[-1]:
            continue
        flag = dataset.loc[ind - window: ind - 1, 'EntryPrice'].mean() - dataset.loc[ind, 'EntryPrice']
        pct_chg = dataset.loc[ind+hold_window, 'EntryPrice'] / dataset.loc[ind, 'EntryPrice'] - 1.0
        if flag < 0:
            ret_pct = -1. * pct_chg
        else:
            ret_pct = pct_chg
        ret_series.append(ret_pct)
        net_value.append(net_value[-1] + initial_cap * ret_pct)

    return pd.Series(net_value, name='net_value')


def spike_accuracy(data, spike_record, base_ret, window=3, file_name=None):
    """
    To find the percentage of spikes occurring in reversion and momentum respectively.
    :param data: pd.DataFrame, contains EntryPrize(vwap), and has integer index.
    :param spike_record: Torch.Tensor, shape: (time_stamps, n_neurons, simulation_time)
    :param base_ret: The baseline return to distinguish spikes.
    :param window: The window length to determine whether the price behavior is reversion or momentum, default 3.
    :param file_name: If not None, save the results to file.
    :return: dictionary: "reversion_pct", "momentum_pct", "total_num", "spike_num", "ascending", "descending"
    """
    dataset = add_spike_column(data=data, spike_record=spike_record)
    tmp = dataset.loc[dataset.spike == 1]
    total_num = len(tmp)
    spike_num = momentum_num = reversion_num = 0.
    ascending = descending = 0.
    for i, ind in enumerate(tmp.index):
        if ind - window < dataset.index[0] or ind + window > dataset.index[-1]:
            continue
        # Note that in pandas.DataFrame.loc, both the start and the stop are included
        ret_after = np.array(dataset.loc[ind+1: ind+window, 'EntryPrice']) / np.array(dataset.loc[ind: ind+window-1, 'EntryPrice']) - 1
        mean_abs_ret = np.nanmean(np.abs(ret_after))
        diff1 = dataset.loc[ind - window: ind - 1, 'EntryPrice'].mean() - dataset.loc[ind, 'EntryPrice']
        diff2 = dataset.loc[ind + 1: ind + window, 'EntryPrice'].mean() - dataset.loc[ind, 'EntryPrice']
        if diff1 < 0:
            ascending += 1
        else:
            descending += 1

        if mean_abs_ret >= base_ret:
            spike_num += 1
            if diff1 * diff2 >= 0:
                reversion_num += 1
            else:
                momentum_num += 1

    acc = {'reversion_pct': reversion_num / spike_num, 'momentum_pct': momentum_num / spike_num,
           'spike_acc': spike_num / total_num, 'total_num': total_num,
           'ascending': ascending / total_num, 'descending': descending / total_num}
    if file_name is not None:
        pass  # TODO save the results to json file.
    return acc


def plot_result(data, spike_record, n=6, display_time=10, file_name=None, plot_every=True, epoch=0):
    """
    Plot the price chart and plot spikes in the corresponding time.
    :param data: pd.DataFrame, should at least have EntryPrice and EntrySize column, and integer index.
    :param spike_record: Torch.Tensor, shape: (time_stamps, n_neurons, simulation_time)
    :param n: To determine the vertical range of the spikes. (use reciprocal)
    """
    dataset = add_spike_column(data=data, spike_record=spike_record)
    y_len = (dataset.EntryPrice.max() - dataset.EntryPrice.min()) / n
    tmp = dataset.loc[dataset.spike == 1]
    if plot_every:
        directory = './images/epoch'+str(epoch)+'/'
        if not os.path.exists(directory):
            os.mkdir(directory)
        else:
            file_list = [os.path.join(directory, f) for f in os.listdir(directory)]
            for f in file_list:
                os.remove(f)
        for i, ind in enumerate(tmp.index):
            plot_aspike(data, ind, file_name=directory+'spike'+str(i)+'.png')

    fig, ax = plt.subplots()
    ax.plot(dataset.index, dataset.EntryPrice)
    ax.vlines(tmp.index, tmp.EntryPrice - y_len, tmp.EntryPrice + y_len, color='r')
    if file_name is not None:
        fig.savefig(file_name)
    plt.pause(display_time)
    plt.close(fig)


def plot_aspike(data, spike_index, plot_range=30, display_time=1, file_name=None):
    """
    Plot one single spike and the nearby price chart.
    :param data: pd.DataFrame containing all the price data(EntryPrice and EntrySize column), and integer index.
    :param spike_index: integer, the spike index.
    :param plot_range: The length of the price that will be shown in the plot.
    :param display_time: The time(seconds) that the plot persists.
    :param file_name: If not None, save figure to file.
    """
    tmp_df = data.loc[(spike_index - plot_range): (spike_index + plot_range), :]
    fig, ax = plt.subplots()
    ax.plot(tmp_df.index, tmp_df.EntryPrice)
    ax.vlines(spike_index, tmp_df.EntryPrice.min(), tmp_df.EntryPrice.max(), color='r')
    if file_name is not None:
        fig.savefig(file_name)
    if display_time is not None:
        plt.pause(display_time)
    plt.close(fig)


class Trainer:
    def __init__(self, net_model, time, single_in=True, print_interval=10):
        self.net_model = net_model
        self.time = time
        self.single_in = single_in
        self.print_interval = print_interval
        self.n_output = self.net_model.n_output
        # Record spikes during the simulation.
        self.train_spike_record = torch.zeros([])  # We will reinitialize train_spike_record in self.training function.
        self.test_spike_record = torch.zeros([])  # We will reinitialize test_spike_record in self.testing function.
        self.monitors = self.net_model.create_monitors(time=self.time)

    def training(self, train_data_loader, n_train, out_layer="Y", plot=False, normalize_weight=True):
        self.train_spike_record = torch.zeros(n_train, self.n_output, self.time)
        print('Begin training.\n')
        start = datetime.now()

        perf_ax = spike_ims = spike_axes = None  # For plot convenience.
        for i in range(n_train):
            if i % self.print_interval == 0:
                print('Progress: %d / %d (%s time)' % (i, n_train, datetime.now() - start))
                start = datetime.now()

            # Get next input sample.
            sample = next(train_data_loader)
            if self.single_in:
                inpts = {'X': sample}
            else:
                inpts = {'X1': sample[:, 0].view(-1, 1), 'X2': sample[:, 1].view(-1, 1)}
            # Run the network on the input.
            self.net_model.run(inpts=inpts, time=self.time)

            # Add to spikes recording.
            self.train_spike_record[i] = self.monitors[out_layer].get('s')
            if self.train_spike_record[i].sum() > 0:
                print(self.train_spike_record[i])
                self.net_model.network.reset_()
            # print(self.monitors[out_layer].get('s'))
            # print(self.monitors[out_layer].get('s').t())

            if normalize_weight:
                self.net_model.network.normalize_weights()
            # Optionally plot various simulation information.
            if plot:
                spike_ims, spike_axes = assembly_plot(self.monitors, spike_ims, spike_axes)
            # self.net_model.network.reset_()   # Here for time series, we shouldnot reset the state variable.

        print('Progress: %d / %d (%s time)\n' % (n_train, n_train, datetime.now() - start))
        print('Training complete.\n')
        return self.train_spike_record.clone()

    def testing(self, test_data_loader, n_test, out_layer="Y", plot=False):
        # Important: set the network to testing mode.
        self.net_model.network.learning = False
        # self.net_model.network.to_testing_mode()  # Alternative method.

        self.test_spike_record = torch.zeros(n_test, self.n_output, self.time)
        print('Begin testing.\n')
        start = datetime.now()

        spike_ims = spike_axes = None  # For plot convenience.
        for i in range(n_test):
            if i % self.print_interval == 0:
                print('Progress: %d / %d (%s time)' % (i, n_test, datetime.now() - start))
                start = datetime.now()

            # Get next input sample.
            sample = next(test_data_loader)
            if self.single_in:
                inpts = {'X': sample}
            else:
                inpts = {'X1': sample[:, 0], 'X2': sample[:, 1]}

            # Run the network on the input.
            self.net_model.run(inpts=inpts, time=self.time)
            # Add to spikes recording.`
            self.test_spike_record[i] = self.monitors[out_layer].get('s')
            if self.test_spike_record[i].sum() > 0:
                print(self.test_spike_record[i])
                self.net_model.network.reset_()

            # Optionally plot various simulation information.
            if plot:
                pspike_ims, spike_axes = assembly_plot(self.monitors,  spike_ims, spike_axes)

        print('Progress: %d / %d (%s time)\n' % (n_test, n_test, datetime.now() - start))
        print('Testing complete.\n')

        return self.test_spike_record.clone()
