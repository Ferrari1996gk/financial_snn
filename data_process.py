#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/4 15:57
# @Author  : Kang
# @Site    : 
# @File    : data_process.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

class IntradayDataHandle:
    def __init__(self, data_path):
        self.raw_data = pd.read_csv(data_path, compression='gzip', skiprows=1)
        self.dominant = self.get_dominant()

    def get_dominant(self):
        """
        Find the dominant contract which has the largest volume.
        :return: The dominant contract trades information.
        """
        dominant_contract = ''
        max_volume = 0
        for contract, df in self.raw_data.groupby('Symbol'):
            total = df.EntrySize.sum()
            if total > max_volume:
                dominant_contract = contract
                max_volume = total
            print('Contract: ' + contract + ' Volume: %d' % total)
        print('The dominant contract is: %s' % dominant_contract)
        dominant = self.raw_data.loc[self.raw_data.Symbol == dominant_contract]
        return dominant.reset_index(drop=True)

    def get_price_series(self, data_type='tensor', length=None):
        """
        :param type: if 'torch', return torch.Tensor; if series, return pd.Series.
        :param length: If not None, return the first length entries.
        Return the entry price in the dominant contracts.
        """
        if length is None:
            length = len(self.dominant)
        if data_type == 'series':
            return self.dominant.EntryPrice.iloc[:length]
        else:
            return torch.Tensor(self.dominant.EntryPrice.iloc[:length].values)


if __name__ == '__main__':
    data_path = './data/201706/Trades/CME.20170601-20170602.F.Trades.382.CL.csv.gz'
    ex = IntradayDataHandle(data_path)
    fig, ax = plt.subplots()
    ax.plot(ex.dominant.EntryPrice.iloc[:5000] - 4780)
    # print(ex.dominant.columns)
    ax.bar([i for i in range(5000)], ex.dominant.EntrySize.iloc[:5000])
    plt.pause(5)
    plt.close(fig)
