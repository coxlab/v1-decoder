
import os
import h5py
import numpy as np
import pandas as pd
import sys
from data_helpers import grouper
from tempConv import tempConvDecoder

offset_arg = int(sys.argv[1])
print('offset: ', offset_arg)
## get and format data
# lfp power bands
# lfp_file = h5py.File('datasets/GRat31_636061_lfp_power.hdf5', 'r')
# lfp_data = np.asarray(lfp_file['lfp_power']) # iterate through powerbands

# sorted spike rates
spikes_file = h5py.File('datasets/GRat31_636061_all_sorted_spikes.hdf5', 'r')
spikes_data = np.array(spikes_file['sorted_spikes'])#[:,2:8]
print('spikes_data shape: ', spikes_data.shape)

# concatenate all neural data
# neural_data = np.concatenate((spikes_data, lfp_data), axis=0)
# print(neural_data.shape)

# tetrodes = grouper(neural_data, neural_data.shape[0])
# print(tetrodes.shape)

head_signals_h5 = h5py.File('datasets/GRat31_636061_all_head_data.hdf5', 'r')
idx_start, idx_stop = [6,9]
head_signals = np.asarray(
    [np.asarray(head_signals_h5[key]) for key in head_signals_h5.keys()][0:9]
).T[:,idx_start:idx_stop]
print('head_signals shape: ', head_signals.shape)

head_signals_keys = list(head_signals_h5.keys())[0:9][idx_start:idx_stop]
head_signals_int = ['yaw_abs', 'roll_abs', 'pitch_abs']
print('head_signals_keys intuitive: ', head_signals_int)

stats = []

for offset in [offset_arg]:
    head_signal = head_signals[:,2]
    id = '{}_{}'.format(0,offset)
    TCD = tempConvDecoder(spikes_data,head_signal,['yaw_abs'],window=30,offset=offset,id=id)
    # TCD = tempConvDecoder(head_signals,head_signal,['yaw_abs'],window=30,offset=offset,id=id)
    TCD.fit()
    R2s,rs = TCD.determine_fit()
    print(offset,R2s,rs)
    stats.append([offset, R2s, rs])

print(stats)
