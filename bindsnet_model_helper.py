# -*- coding: utf-8 -*-
# @Time    : 2019/6/23 15:31
# @Author  : Kang
# @File    : bindsnet_model_helper.py
# @Software: PyCharm
import torch
from bindsnet.network import Network
from bindsnet.learning import PostPre, NoOp
from bindsnet.network.topology import Connection, LocallyConnectedConnection
from bindsnet.network.nodes import Input, RealInput, LIFNodes, DiehlAndCookNodes


class FeedForwardNetwork(Network):
    # language=rst
    """
    Implements an ``Input`` instance connected to a ``LIFNodes`` instance with a fully-connected ``Connection``.
    """

    def __init__(self, n_inpt, n_neurons=64, n_output=1, dt=1.0, wmin=0.0, wmax=1.0, nu=(1e-4, 1e-2), norm=78.4):
        # language=rst
        """
        Constructor for class ``TwoLayerNetwork``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of neurons in the hidden layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param wmin: Minimum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param wmax: Maximum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param norm: ``Input`` to ``LIFNodes`` layer connection weights normalization constant.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.n_neurons = n_neurons
        self.n_output = n_output
        self.dt = dt

        self.add_layer(Input(n=self.n_inpt, traces=True, trace_tc=5e-2), name='X')
        self.add_layer(LIFNodes(n=self.n_neurons, traces=True, rest=-65.0, reset=-65.0, thresh=-52.0, refrac=5,
                                decay=1e-2, trace_tc=5e-2), name='H')
        self.add_layer(LIFNodes(n=1, traces=True, rest=-65.0, reset=-65.0, thresh=-52.0, refrac=5,
                                decay=1e-2, trace_tc=5e-2), name='Y')

        w1 = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        w2 = 0.3 * torch.rand(self.n_neurons, self.n_output)
        self.add_connection(Connection(source=self.layers['X'], target=self.layers['H'], w=w1, update_rule=PostPre,
                                       nu=nu, wmin=wmin, wmax=wmax, norm=norm), source='X', target='H')
        self.add_connection(Connection(source=self.layers['H'], target=self.layers['Y'], w=w2, update_rule=PostPre,
                                       nu=nu, wmin=wmin, wmax=wmax, norm=norm), source='H', target='Y')

    def to_testing_mode(self):
        """
        When called, change the network to testing mode, that is, change the learning rule of the connections to NoOp
        so that the weights are fixed.
        :return: No return, modify the network in place.
        """
        keys_list = list(self.connections.keys())
        for key in keys_list:
            tmp_conn = self.connections.pop(key)
            w, nu, wmin, wmax, norm = tmp_conn.w, tmp_conn.nu, tmp_conn.wmin, tmp_conn.wmax, tmp_conn.norm
            self.add_connection(Connection(source=self.layers[key[0]], target=self.layers[key[1]], w=w, update_rule=NoOp,
                                           nu=nu, wmin=wmin, wmax=wmax, norm=norm), source=key[0], target=key[1])



