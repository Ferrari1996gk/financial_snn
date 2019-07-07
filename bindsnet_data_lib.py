#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/19 20:00
# @Author  : Kang
# @Site    : 
# @File    : bindsnet_data_lib.py
# @Software: PyCharm
import torch
import numpy as np
from data_process import IntradayDataHandle
from bindsnet.encoding import poisson_loader, bernoulli_loader


def get_model_data(data_path, data_type='tensor', length=None):
    # Utilise DataHandle class to get raw_model_data.
    print("Begin to prepare raw model data from excel!")
    ex = IntradayDataHandle(data_path=data_path)
    if length is None:
        return ex.get_price_series(data_type=data_type, length=length), ex.dominant
    else:
        return ex.get_price_series(data_type=data_type, length=length), ex.dominant.iloc[:length]


def train_test_split(raw_model_data, test_percent=None, n_train=None, n_test=None,
                     normal=False, new_mean=None, new_std=None):
    """
    Split the dataset into training set and testing set. Also transform the numpy array to torch Tensor.
    :param raw_model_data: The original data, numpy.array
    :param test_percent: The percentage of testing set.
    :param n_train: Number of train set instances. If given, n_test must be given at the same time.
    :param n_test: Number of testing set instances.
    :param normal: If True, normalize the data, using the new mean value and the new standard deviation.
    :param new_mean: The new mean value.
    :param new_std：The new data standard deviation after normalize.
    :return: train_series, test_series. All these are torch.Tensor
    """
    model_data = raw_model_data.clone()
    if n_train is not None:
        assert n_test is not None, 'If you give n_train you must give n_test!'
        train_data = model_data[:n_train]
        test_data = model_data[n_train:(n_train + n_test)]
    else:
        assert test_percent is not None, 'A test set percentage is needed. Otherwise give training and testing size!'
        split = int(len(model_data) * (1 - test_percent))
        train_data = model_data[:split]
        test_data = model_data[split:]

    if normal:
        assert new_mean is not None and new_std is not None, "Normalize the data needs mean value and data range."
        processor = PreProcessor(train_data)
        train_data = processor.apply(train_data, new_mean, new_std)
        test_data = processor.apply(test_data)
    return train_data, test_data


class PreProcessor:
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """
    def __init__(self, train_data):
        """
        :param train_data: torch.Tensor, 1 - dimension.
        """
        self.mean = train_data.mean()
        self.stdev = train_data.std()
        self.new_mean = self.new_std = None

    def apply(self, data, new_mean=None, new_std=None):
        """
        Normalize the data to a certain mean and standard deviation.
        :param data: torch.Tensor, 1 - dimension.
        """
        if new_mean is not None:
            self.new_mean = new_mean
            self.new_std = new_std
            return (data - self.mean) / self.stdev * new_std + new_mean
        else:
            assert self.new_mean is not None, "Please initialize new_mean and new_std!"
            return (data - self.mean) / self.stdev * self.new_std + self.new_mean

    def revert(self, data):
        """
        Revert the pre-processing operations to retreive the original dataset.
        :param data: dataset for which to revert normalization.
        """
        assert self.new_mean is not None, "You haven't done normalization!"
        return (data - self.new_mean) / self.new_std * self.stdev + self.mean


def bindsnet_load_data(dataset, time, dt, method='poisson'):
    """
    Generates spike trains based on input intensity, encoding a sequence of data.
    The methods currently can be "poisson" or "bernoulli"
    :param dataset: The dataset to be encoded as spike trains.
    :param time: Length of spike train per input variable.
    :param dt: Simulation time step.
    :param method: "poisson" or "bernoulli".
    :return: The spikes encoded by the assigned method. type: torch.Tensor
    """
    if method == 'poisson':
        data_loader = poisson_loader(data=dataset, time=time, dt=dt)
    elif method == 'bernoulli':
        data_loader = bernoulli_loader(data=dataset, time=time, dt=dt)
    else:
        raise Exception("You need to give a correct encoding method(\"poisson\" or \"bernoulli\")!")
    return data_loader
