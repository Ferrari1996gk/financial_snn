# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:02:14 2019
@author: kang
"""
import torch
from bindsnet_data_lib import get_model_data, train_test_split, bindsnet_load_data
from bindsnet_train_lib import Trainer, plot_result
from bindsnet_models import DiehlCookModel2015Version2, DiehlCookModel2015, TwoLayerModel, MultiLayerModel

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
n_neurons = 64; test_percent = 0.7; n_total = 10000
epoch = 1
exc = 0.5; inh = 1.0; time = 6; dt = 1.0
method = "poisson"

# Intra-day high frequency data
# For data pre-processing
new_mean = 70.0
new_std = 10.0
data_path = './data/201706/Trades/CME.20170531-20170601.F.Trades.382.CL.csv.gz'
raw_model_data, dominant = get_model_data(data_path=data_path, data_type='tensor', length=n_total)
raw_model_data = torch.cat([torch.Tensor([0]), raw_model_data[1:] - raw_model_data[:-1]])

train_data, test_data = train_test_split(raw_model_data, test_percent=test_percent, n_train=None, n_test=None,
                                         normal=True, new_mean=new_mean, new_std=new_std)
print(train_data[train_data < 0])
train_data[train_data < 0] = 0
test_data[test_data < 0] = 0

train_data = train_data.repeat(window, 1).transpose(0,1)
test_data = test_data.repeat(window, 1).transpose(0,1)
print(train_data.shape)
n_train = len(train_data)
# Build network model.
out_layer = 'Y'
# net = DiehlCookModel2015(n_inpt=window, n_neurons=n_neurons, exc=exc,
#         inh=inh, dt=dt, nu=[0, 0.25], wmin=0, wmax=5, norm=100, theta_plus=0.02)
# print('DiehlCook2015 model!')
# net = TwoLayerModel(n_inpt=window, n_neurons= n_neurons, dt=dt, initial_w=0.1, wmin=0.0, wmax=1.0, nu= (0.05, 0.05), norm=0.65)
# print('Two layer model!')
# net = DiehlCookModel2015Version2(n_inpt=window, n_neurons=n_neurons, inh=inh, dt=dt, nu=[0.05, 0.05],
#                                  wmin=0, wmax=5, norm=100, theta_plus=0.2)
# print('DiehlCook2015 model version 2!')
net = MultiLayerModel(n_inpt=window, n_neurons=n_neurons, n_output=1, dt=dt, initial_w=10, wmin=None, wmax=None, nu=(1, 1), norm=512)
print('MultiLayerModel, hidden size = %d!' % n_neurons)
ex = Trainer(net, time, print_interval=print_interval)
before = ex.net_model.network.connections[('X', 'H')].w.clone().numpy()
print(before)
for i in range(epoch):
    print('Training epoch number: %d' % (i + 1))
    # Lazily encode data as spike trains.
    train_data_loader = bindsnet_load_data(dataset=train_data, time=time, dt=dt, method=method)
    train_record = ex.training(train_data_loader, n_train, out_layer=out_layer, plot=plot, normalize_weight=True)
    plot_result(data=dominant.iloc[:n_train], spike_record=train_record, display_time=10, file_name='./images/epoch'+str(i)+'.png',
                plot_every=True, epoch=i)
    ex.net_model.network.reset_()
middle = ex.net_model.network.connections[('X', 'H')].w.clone().numpy()
print(middle)

##############The following is for testing####################################################
print('Training finished, begin testing!!!!!!')
n_test = len(test_data)
test_data_loader = bindsnet_load_data(dataset=test_data, time=time, dt=dt, method=method)
test_record = ex.testing(test_data_loader, n_test, out_layer=out_layer, plot=plot)
plot_result(data=dominant.iloc[n_train:n_train+n_test], spike_record=test_record, display_time=10, file_name=None,
                plot_every=True)
after = ex.net_model.network.connections[('X', 'H')].w.clone().numpy()
print(after)
print(before == middle)
print(middle == after)