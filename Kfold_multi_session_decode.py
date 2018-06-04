
# coding: utf-8

# In[9]:

import os
import gc
from pympler import tracker
from datetime import datetime
from sklearn.model_selection import KFold
from data_helpers import format_timeseries
from tempConv import tempConvDecoder

tr = tracker.SummaryTracker()

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
    'X_split' : (0,384,24),
    'X_frac' : .4,
    'y_split' : (7,8),
    'y_frac' : .5, 
    'y_names' : ['absolute pitch'],
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

def sample_train_test(train_paths, test_paths, conf):
    X_train, y_train = format_timeseries(
        train_paths, 
        conf['window'],
        conf['offset'],
        conf['X_split'], 
        conf['y_split'], 
        sample_frac=conf['X_frac']
    )
    print('sample train loaded: %s' % train_paths)
    X_test, y_test = format_timeseries(
        test_paths, 
        conf['window'],
        conf['offset'],
        conf['X_split'], 
        conf['y_split'], 
        sample_frac=conf['y_frac']
    )
    print('sample train loaded: %s' % train_paths)
    
    return X_train, y_train, X_test, y_test

# split dataset_paths, n-1 to train, 1 to test
kf = KFold(n_splits=len(dataset_paths))
for train_paths, test_paths in kf.split(dataset_paths):
    print("train on:\n %s\n test on:\n %s\n" % (train_paths, test_paths))
    
    # collect statistics from each epoch
    stats = []
    
    
    ## train for number of epochs, resampling training datasets at each new epoch
    for epoch in range(conf['eps']):
        #### train model and test model for each Kfold
        # sample first round of data:
        X_train, y_train, X_test, y_test = [None, None, None, None] # delete residual data first
        X_train, y_train, X_test, y_test = sample_train_test(train_paths, test_paths, conf)

        if epoch == 0:
            conf['input_shape'] = X_train.shape
            conf['output_shape'] = y_train.shape

            ## define model on first epoch
            TCD = tempConvDecoder(**conf)

        print('epoch:', epoch)
        TCD.model.fit(
            X_train,
            y_train,
            epochs=1, 
            batch_size=conf['bs']
        )
        
        R2s,rs = TCD.determine_fit(X_test, y_test)
        stats.append([R2s, rs])
    
        print("R2: %s\n r: %s" % R2s, rs)
        gc.collect()
        tr.print_diff()
    
    print(stats)
