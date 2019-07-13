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

    def __init__(self, n_inpt, n_neurons=64, n_output=1, dt=1.0, wmin=None, wmax=None, nu=(1, 1), norm=512):
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
        self.add_layer(LIFNodes(n=self.n_neurons, traces=True, rest=-65.0, reset=-65.0, thresh=-52.0, refrac=50,
                                decay=2, trace_tc=5e-2), name='H')
        self.add_layer(LIFNodes(n=1, traces=True, rest=-65.0, reset=-65.0, thresh=-52.0, refrac=50,
                                decay=2, trace_tc=5e-2), name='Y')

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

    def normalize_weights(self):
        """
        Normalize the connection weights from 'H' to 'Y'
        """
        self.connections[('H', 'Y')].normalize()


class DoubleInputNetwork(Network):
    # language=rst
    """
    Implements a feed forward network with double inputs and local connection.
    """

    def __init__(self, n_neurons=64, n_output=1, dt=1.0, initial_w=10, wmin=None, wmax=None, nu=(1, 1), norm=512):
        # language=rst
        """
        Constructor for class ``TwoLayerNetwork``.

        :param n_neurons: Number of neurons in the hidden layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param wmin: Minimum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param wmax: Maximum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param norm: ``Input`` to ``LIFNodes`` layer connection weights normalization constant.
        """
        super().__init__(dt=dt)

        self.n_neurons = n_neurons
        self.n_output = n_output
        self.dt = dt

        self.add_layer(Input(n=1, traces=True, trace_tc=5e-2), name='X1')
        self.add_layer(Input(n=1, traces=True, trace_tc=5e-2), name='X2')
        self.add_layer(LIFNodes(n=self.n_neurons, traces=True, rest=-65.0, reset=-65.0, thresh=-52.0, refrac=50,
                                decay=2, trace_tc=5e-2), name='H1')
        self.add_layer(LIFNodes(n=self.n_neurons, traces=True, rest=-65.0, reset=-65.0, thresh=-52.0, refrac=50,
                                decay=2, trace_tc=5e-2), name='H2')
        self.add_layer(LIFNodes(n=1, traces=True, rest=-65.0, reset=-65.0, thresh=-52.0, refrac=50,
                                decay=2, trace_tc=5e-2), name='Y')

        w1 = initial_w * torch.rand(1, self.n_neurons)
        w2 = initial_w * torch.rand(1, self.n_neurons)
        w3 = initial_w * torch.rand(self.n_neurons, self.n_output)
        w4 = initial_w * torch.rand(self.n_neurons, self.n_output)
        self.add_connection(Connection(source=self.layers['X1'], target=self.layers['H1'], w=w1, update_rule=PostPre,
                                       nu=nu, wmin=wmin, wmax=wmax, norm=norm), source='X1', target='H1')
        self.add_connection(Connection(source=self.layers['X2'], target=self.layers['H2'], w=w2, update_rule=PostPre,
                                       nu=nu, wmin=wmin, wmax=wmax, norm=norm), source='X2', target='H2')
        self.add_connection(Connection(source=self.layers['H1'], target=self.layers['Y'], w=w3, update_rule=PostPre,
                                       nu=nu, wmin=wmin, wmax=wmax, norm=norm), source='H1', target='Y')
        self.add_connection(Connection(source=self.layers['H2'], target=self.layers['Y'], w=w4, update_rule=PostPre,
                                       nu=nu, wmin=wmin, wmax=wmax, norm=norm), source='H2', target='Y')

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

    def normalize_weights(self):
        """
        Normalize the connection weights from 'H' to 'Y'
        """
        self.connections[('H1', 'Y')].normalize()
        self.connections[('H2', 'Y')].normalize()
