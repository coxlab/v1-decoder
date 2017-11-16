
# coding: utf-8

# In[2]:

import os
import h5py
import numpy as np
import pandas as pd
from data_helpers import grouper
from tempConv import tempConvDecoder

# In[3]:

## get and format data
# lfp power bands
lfp_file = h5py.File('datasets/synth_X', 'r')
lfp_data = np.asarray(lfp_file['X'])[:,0:2] # iterate through powerbands
print(lfp_data.shape)


# In[4]:

head_signals_h5 = h5py.File('datasets/synth_y', 'r')
head_signal = np.array(head_signals_h5['y'])
print('head_signals shape: ', head_signal.shape)


# In[9]:

stats = []


# In[ ]:

for offset in range(-46,41,3):
    TCD = tempConvDecoder(lfp_data,head_signal,['yaw_abs'],window=30,offset=offset)
    TCD.fit()
    R2s,rs = TCD.determine_fit()
    stats.append([[offset], R2s, rs])


# In[ ]:

print(stats)

