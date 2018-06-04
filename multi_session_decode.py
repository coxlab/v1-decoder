
# coding: utf-8

# In[9]:

import os
import gc
from pympler import tracker
from datetime import datetime
from data_helpers import format_timeseries
from tempConv import tempConvDecoder

# In[20]:

conf = {
    'bs' : 512,
    'eps' : 8,
    'lr' : 0.0005,
    'kernel' : 2,
    'nb_filter' : 5,
    'window' : 60,
    'offset' : 20,
    'regressor' : True,
    'pyramidal' : True,
    'resample_data' : False,
    'sample_size' : 40000,
    'run_id' : datetime.now(),
    'verbose' : False,
    'key' : ['absolute pitch'],
    'input_shape' : None,
    'output_shape' : None
}


# In[13]:

## produce dataset_paths, format: [[path/to/X1, path/to/y1], [path/to/X2, path/to/y2]..]
## example of collecting some datasets into a list of tuples
rat_path = 'datasets/GRat31/'
data_keys = [folder for folder in os.listdir('datasets/GRat31/')]
folders = [rat_path+folder for folder in os.listdir('datasets/GRat31/')]

X_fname = 'lfp_power.hdf5'
y_fname = 'all_head_data.hdf5'
dataset_paths = []
for folder in folders:
    data_file_list = os.listdir(folder)
    dataset_paths.append([
        folder+'/'+data_file_list[data_file_list.index(X_fname)],
        folder+'/'+data_file_list[data_file_list.index(y_fname)]
    ])

# lets train on the fifth dataset and test on the rest 
# test_paths = [dataset_paths.pop(4)]
dataset_paths.pop(4)
test_paths = [dataset_paths.pop(3)]
# test_paths = [dataset_paths[3]]

tr = tracker.SummaryTracker()
X_train, y_train = format_timeseries(
    dataset_paths, 
    conf['window'],
    conf['offset'],
    (0,384,24), 
    (7,8),
    sample_frac=.4
)
print('loaded train')
X_test, y_test = format_timeseries(
    test_paths, 
    conf['window'],
    conf['offset'],
    (0,384,24), 
    (7,8),
    sample_frac=.5
)
print('loaded test')

# import pdb; pdb.set_trace()

tr.print_diff()
conf['input_shape'] = X_train.shape
conf['output_shape'] = y_train.shape
# In[15]:

stats = []


# In[18]:

## define model
TCD = tempConvDecoder(**conf)

## train for number of epochs, resampling training datasets at each new epoch
for _ in range(conf['eps']):
    print('epoch:', _)
    TCD.model.fit(
        X_train,
        y_train,
        epochs=1, 
        batch_size=conf['bs']
    )
    X_train, y_train = format_timeseries(
        dataset_paths, 
        conf['window'],
        conf['offset'],
        (0,384,24), 
        (7,8),
        sample_frac=.4
    )
    X_test, y_test = format_timeseries(
        test_paths, 
        conf['window'],
        conf['offset'],
        (0,384,24), 
        (7,8),
        sample_frac=.5
    )
    R2s,rs = TCD.determine_fit(X_test, y_test)
    stats.append([R2s, rs])

    print(stats)
    gc.collect()
    tr.print_diff()

print(stats)
