import os
import h5py
import numpy as np
import pandas as pd
import sys
from data_helpers import grouper
from tempConv import tempConvDecoder

f = open(sys.argv[0])
print('python script contents:')
for line in f:
    print(line)

## get and format data
folder = '/n/home11/guitchounts/ephys/GRat31/636427282621202061/'
# 'all_head_data_100hz, mua_firing_rates_100hz.hdf5'
# sorted spike rates
spikes_file = h5py.File(folder+'mua_firing_rates_100hz.hdf5', 'r')
spikes_data = np.asarray(spikes_file['firing_rates'])
print('spikes_data shape: ', spikes_data.shape)

head_signals_h5 = h5py.File(folder+'all_head_data_100hz.hdf5', 'r')
idx_start, idx_stop = [6,9]
head_signals = np.asarray(
    [np.asarray(head_signals_h5[key]) for key in head_signals_h5.keys()][0:9]
).T[:,idx_start:idx_stop]
print('head_signals shape: ', head_signals.shape)

head_signals_keys = list(head_signals_h5.keys())[0:9][idx_start:idx_stop]
print("head_signals_keys: ", head_signals_keys)
head_signals_int = ['yaw_abs', 'roll_abs', 'pitch_abs']
print('head_signals_keys intuitive: ', head_signals_int)

stats = {}

# iterate Xs
for run_idx in range(3): #range(tetrodes.shape[0]):
    # tetrode = tetrodes[tetrode_idx].T
    # print('>>> t: ',tetrode.shape)
    # if tetrode_idx >= 1: break

    # iterate ys
    for head_signal_idx in range(head_signals.shape[1]):
        R2r_arr = {
            'R2s' : [],
            'rs' : []
        }

        for offset in [-4000,-2000,-500,-10,-5,0,5,10,500,2000,4000]:
            head_signal = head_signals[:,head_signal_idx]
            id = '{}_{}'.format(head_signal_idx,offset)
            TCD = tempConvDecoder(spikes_data,head_signal,['yaw_abs'],window=300,offset=offset,id=id)
            # TCD = tempConvDecoder(,head_signal,['yaw_abs'],window=30,offset=10,id=id, percent_data=p)
            TCD.fit()
            R2s,rs = TCD.determine_fit()
            print(offset,R2s,rs)
            R2r_arr['R2s'].append(R2s)
            R2r_arr['rs'].append(rs)

        stats['run_{}_head_signal_{}'.format(run_idx, head_signal_idx)] = R2r_arr


# In[ ]:

print(stats)
