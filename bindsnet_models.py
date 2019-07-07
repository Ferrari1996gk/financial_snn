#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/21 21:04
# @Author  : Kang
# @Site    : 
# @File    : bindsnet_models.py
# @Software: PyCharm
import torch
import numpy as np
from bindsnet_model_helper import FeedForwardNetwork
from bindsnet.models import DiehlAndCook2015, TwoLayerNetwork, DiehlAndCook2015v2
from bindsnet.network.monitors import Monitor


class BaseModel:
    """
    Base model class. All our models will inherit this class.
    The input layer of all the network should be named 'X'
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def create_monitors(self, *args, **kwargs):
        raise NotImplementedError


class DiehlCookModel2015(BaseModel):
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(self, n_inpt, n_neurons=100, exc=22.5, inh=17.5, dt=1.0, nu=(1e-4, 1e-2), wmin=0.0, wmax=1.0,
                 norm=78.4, theta_plus=0.05, theta_decay=1e-7, X_Ae_decay=None, Ae_Ai_decay=None, Ai_Ae_decay=None):
        self.n_neurons = n_neurons
        self.n_output = n_neurons
        self.network = DiehlAndCook2015(n_inpt=n_inpt, n_neurons=n_neurons, exc=exc, inh=inh, dt=dt, nu=nu, wmin=wmin,
                                        wmax=wmax, norm=norm, theta_plus=theta_plus, theta_decay=theta_decay,
                                        X_Ae_decay=X_Ae_decay, Ae_Ai_decay=Ae_Ai_decay, Ai_Ae_decay=Ai_Ae_decay)

    def run(self, inpts, time, clamp=None):
        if clamp is not None:
            self.network.run(inpts=inpts, time=time, clamp=clamp)
        else:
            self.network.run(inpts=inpts, time=time)
        
    def create_monitors(self, time):
        monitors = {}
        for layer in set(self.network.layers) - {'X'}:
            monitors[layer] = Monitor(self.network.layers[layer], state_vars=['s'], time=time)
            self.network.add_monitor(monitors[layer], name='%s_monitor' % layer)
        return monitors


class TwoLayerModel(BaseModel):
    def __init__(self, n_inpt, n_neurons= 100, dt=1.0, initial_w=0.1, wmin=0.0, wmax=1.0, nu= (1e-4, 1e-2), norm=78.4):
        self.n_inpt = n_inpt
        self.n_neurons = n_neurons
        self.n_output = n_neurons
        self.network = TwoLayerNetwork(n_inpt=n_inpt, n_neurons=n_neurons, dt=dt, wmin=wmin, wmax=wmax, nu=nu, norm=norm)
        self.network.connections[('X', 'Y')].w = initial_w * torch.rand(self.n_inpt, self.n_neurons)

    def run(self, inpts, time, clamp=None):
        if clamp is not None:
            self.network.run(inpts=inpts, time=time, clamp=clamp)
        else:
            self.network.run(inpts=inpts, time=time)

    def create_monitors(self, time):
        monitors = {}
        for layer in set(self.network.layers) - {'X'}:
            monitors[layer] = Monitor(self.network.layers[layer], state_vars=['s'], time=time)
            self.network.add_monitor(monitors[layer], name='%s_monitor' % layer)
        return monitors


class DiehlCookModel2015Version2(BaseModel):
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(self, n_inpt, n_neurons=100, inh=17.5, dt=1.0, nu=(1e-4, 1e-2), wmin=0.0, wmax=1.0,
                 norm=78.4, theta_plus=0.05, theta_decay=1e-7):
        self.n_neurons = n_neurons
        self.n_output = n_neurons
        self.network = DiehlAndCook2015v2(n_inpt=n_inpt, n_neurons=n_neurons, inh=inh, dt=dt, nu=nu, wmin=wmin,
                                        wmax=wmax, norm=norm, theta_plus=theta_plus, theta_decay=theta_decay)

    def run(self, inpts, time, clamp=None):
        if clamp is not None:
            self.network.run(inpts=inpts, time=time, clamp=clamp)
        else:
            self.network.run(inpts=inpts, time=time)

    def create_monitors(self, time):
        monitors = {}
        for layer in set(self.network.layers) - {'X'}:
            monitors[layer] = Monitor(self.network.layers[layer], state_vars=['s'], time=time)
            self.network.add_monitor(monitors[layer], name='%s_monitor' % layer)
        return monitors


class MultiLayerModel(BaseModel):
    def __init__(self, n_inpt, n_neurons=64, n_output=1, dt=1.0, initial_w=0.1, wmin=0.0, wmax=1.0, nu=(1e-4, 1e-2), norm=78.4):
        self.n_inpt = n_inpt
        self.n_neurons = n_neurons
        self.n_output = n_output
        self.network = FeedForwardNetwork(n_inpt=n_inpt, n_neurons=n_neurons, n_output=n_output, dt=dt, wmin=wmin,
                                          wmax=wmax, nu=nu, norm=norm)
        self.network.connections[('X', 'H')].w = initial_w * torch.rand(self.n_inpt, self.n_neurons)
        self.network.connections[('H', 'Y')].w = initial_w * torch.rand(self.n_neurons, self.n_output)

    def run(self, inpts, time, clamp=None):
        if clamp is not None:
            self.network.run(inpts=inpts, time=time, clamp=clamp)
        else:
            # print(self.network.connections[('H', 'Y')].w.numpy()[:2])
            self.network.run(inpts=inpts, time=time)

    def create_monitors(self, time):
        monitors = {}
        for layer in set(self.network.layers) - {'X'}:
            monitors[layer] = Monitor(self.network.layers[layer], state_vars=['s'], time=time)
            self.network.add_monitor(monitors[layer], name='%s_monitor' % layer)
        return monitors
