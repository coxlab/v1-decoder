
# coding: utf-8

# In[9]:

import os
import gc
import sys
import numpy as np
import json
from pympler import tracker
from datetime import datetime
from sklearn.model_selection import KFold
from data_helpers import sample_train_test, load_data
from tempConv import tempConvDecoder
from other import get_sha
tr = tracker.SummaryTracker()

conf_path = os.path.abspath(sys.argv[1]) # pass path to config here as arg
rat_path = os.path.dirname(conf_path)
conf = json.load(open(conf_path, 'r'))
conf['git_version'] = get_sha()

# set date file types, ie 'lfp_power.hdf5', from config
X_fname = conf['config']['neural_data']
y_fname = conf['config']['head_data']

# get all folders
data_keys = [folder for folder in os.listdir(rat_path)]
folders = [os.path.join(rat_path,folder) for folder in os.listdir(rat_path)]
folders = filter(lambda x: os.path.isdir(x), folders)

# specify training set (dark)
train_paths = []
test_paths = []

for folder in folders:
    data_file_list = os.listdir(folder)
    # only take folder containing the desire neural data
    if True in [conf['config']['neural_data'] in fil for fil in data_file_list]:
        # only add to train if (dark|light)
        if True in ['m_'+conf['config']['condition'] in fil for fil in data_file_list]:
            train_paths.append([
                folder+'/'+data_file_list[data_file_list.index(X_fname)],
                folder+'/'+data_file_list[data_file_list.index(y_fname)]
            ])
        # only add to train if (light|dark)
        if True in ['m_'+conf['config']['test'] in fil for fil in data_file_list]:
            test_paths.append([
                folder+'/'+data_file_list[data_file_list.index(X_fname)],
                folder+'/'+data_file_list[data_file_list.index(y_fname)]
            ])

train_paths = np.array(train_paths)
test_paths = np.array(test_paths)

print("train on:\n %s\n test on:\n %s\n" % (train_paths, test_paths))

conf_nn = conf['nn_params']
conf_nn['run_id'] = str(datetime.now().timestamp())
conf_nn['train_paths'] = train_paths.tolist()
conf_nn['test_paths'] = test_paths.tolist()
save_path = os.path.join(rat_path, conf['config']['experiment'], conf_nn['run_id'])
if not os.path.exists(save_path):
    print('create save directory: ', save_path)
    os.makedirs(save_path)
conf_nn['save_path'] = save_path

# import pdb; pdb.set_trace()   # uncomment for interactive debugging 

# collect statistics from each epoch
stats = []

## train for number of epochs, resampling training datasets at each new epoch
for epoch in range(conf_nn['eps']):
    #### train model and test model for each Kfold
    # sample first round of data:
    X_train, y_train = load_data(train_paths, conf_nn, sample_frac=conf_nn['X_frac'])

    if epoch == 0:
        conf_nn['input_shape'] = X_train.shape
        conf_nn['output_shape'] = y_train.shape

        ## define model on first epoch
        TCD = tempConvDecoder(**conf_nn)

    print('epoch: %s/%s' % (epoch, conf_nn['eps']))

    TCD.model.fit(
        X_train,
        y_train,
        epochs=1, # ignore, we control epochs with for loop  
        batch_size=conf_nn['bs']
    )
    
    X_train, y_train = [None, None] # delete residual data first
    
    for test_path in test_paths:
        X_test, y_test = load_data([test_path], conf_nn, sample_frac=1)
        # ugly plot results if final epoch
        if epoch+1 == int(conf_nn['eps']):
            R2s,rs = TCD.determine_fit(X_test, y_test, save_result=True)
        else:
            R2s,rs = TCD.determine_fit(X_test, y_test, save_result=False)

        X_test, y_test = [None, None] # delete residual data first
        
        stats.append([R2s, rs])
        print("R2: %s\n r: %s" % (R2s, rs))

    gc.collect()
    tr.print_diff()

conf_nn['stats'] = stats

# save stats and settings after last epoch
with open(os.path.join(save_path, 'trial_info.json'), 'w') as outfile:
    json.dump(conf, outfile)

print(stats)
