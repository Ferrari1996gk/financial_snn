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
    def __init__(self, data_path, vwap_window=10):
        self.raw_data = pd.read_csv(data_path, compression='gzip', skiprows=1)
        # self.dominant = self.get_dominant()
        self.dominant = self.get_dominant_vwap(vwap_window=vwap_window)

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
    
    def get_dominant_vwap(self, vwap_window=10):
        """
        Calculate the vwap price for dominant contract.
        :param vwap_window: The window length for calculating vwap.
        :return: pd.DataFrame, having vwap price in column 'EntryPrice'.
        """
        dominant = self.get_dominant()
        tmp_df = dominant[['EntryPrice', 'EntrySize']].copy()
        tmp_df['group_ind'] = tmp_df.index // vwap_window
        tmp_df['money'] = tmp_df.EntryPrice * tmp_df.EntrySize

        df = tmp_df.groupby('group_ind').sum()
        df['EntryPrice'] = df['money'] / df['EntrySize']
        return df
    
    def get_price_series(self, data_type='price', length=None):
        """
        :param data_type: if 'price', return price data; if 'volume', return volume data.
        :param length: If not None, return the first length entries.
        Return the entry price or volume in the dominant contracts, with torch.Tensor form.
        """
        if length is None:
            length = len(self.dominant)
        if data_type == 'price':
            return torch.Tensor(self.dominant.EntryPrice.iloc[:length].values)
        elif data_type == 'volume':
            return torch.Tensor(self.dominant.EntrySize.iloc[:length].values)
        else:
            raise ValueError("Unknown data type!!!")


if __name__ == '__main__':
    """
    data_path = './data/201705/Trades/CME.20170504-20170505.F.Trades.382.CL.csv.gz'
    ex = IntradayDataHandle(data_path)
    dominant = ex.dominant
    volume = ex.get_price_series(data_type='volume', length=1000)
    fig, ax = plt.subplots()
    ax.plot(dominant.EntryPrice.iloc[:8000] - 4780)
    # print(dominant.columns)
    ax.bar([i for i in range(8000)], ex.dominant.EntrySize.iloc[:8000])
    plt.pause(5)
    plt.close(fig)
    """
    import os
    commodity = 'GC'
    directory = './data/201705/Trades/'
    file_list = os.listdir(directory)
    data_list = [x for x in file_list if x[-9:-7] == commodity]
    volatility = []
    for file_name in data_list:
        date_str = file_name[13:21]
        data_path = directory + file_name
        ex = IntradayDataHandle(data_path)
        dominant = ex.dominant
        price = dominant.EntryPrice
        ret = np.log(price.shift(-1)) - np.log(price)
        ret = ret.dropna()
        vol = np.sum(ret**2)**0.5 * np.sqrt(252)
        print(date_str, ': ', vol)
        volatility.append(vol)
    print('average volatility: %.4f' % np.mean(volatility))
