
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
from data_helpers import sample_train_test
from tempConv import tempConvDecoder
tr = tracker.SummaryTracker()

conf_path = os.path.abspath(sys.argv[1]) # pass path to config here as arg
rat_path = os.path.dirname(conf_path)
conf = json.load(open(conf_path, 'r'))

data_keys = [folder for folder in os.listdir(rat_path)]
folders = [os.path.join(rat_path,folder) for folder in os.listdir(rat_path)]
folders = filter(lambda x: os.path.isdir(x), folders)

X_fname = conf['config']['neural_data']
y_fname = conf['config']['head_data']
dataset_paths = []
            
## produce dataset_paths, format: [[path/to/X1, path/to/y1], [path/to/X2, path/to/y2]..]
## example of collecting some datasets into a list of tuples
for folder in folders:
    data_file_list = os.listdir(folder)
    if True in [conf['config']['condition'] in fil for fil in data_file_list]:
        dataset_paths.append([
            folder+'/'+data_file_list[data_file_list.index(X_fname)],
            folder+'/'+data_file_list[data_file_list.index(y_fname)]
        ])

dataset_paths = np.array(dataset_paths)
dataset_paths

# kfold split dataset_paths, n-1 to train, 1 to test
kf = KFold(n_splits=len(dataset_paths))
for train_idx, test_idx in kf.split(dataset_paths):
    train_paths = dataset_paths[train_idx]
    test_paths = dataset_paths[test_idx]
    print("train on:\n %s\n test on:\n %s\n" % (train_paths, test_paths))

    conf_nn = conf['nn_params']
    conf_nn['run_id'] = str(datetime.now().timestamp())
    conf_nn['train_paths'] = train_paths.tolist()
    conf_nn['test_paths'] = test_paths.tolist()
    save_path = os.path.join(rat_path, conf['config']['experiment'], conf_nn['run_id'])
    conf_nn['save_path'] = save_path
    
    # import pdb; pdb.set_trace()    
    
    # collect statistics from each epoch
    stats = []
    
    ## train for number of epochs, resampling training datasets at each new epoch
    for epoch in range(conf_nn['eps']):
        #### train model and test model for each Kfold
        # sample first round of data:
        X_train, y_train, X_test, y_test = [None, None, None, None] # delete residual data first
        X_train, y_train, X_test, y_test = sample_train_test(train_paths, test_paths, conf_nn)

        if epoch == 0:
            conf_nn['input_shape'] = X_train.shape
            conf_nn['output_shape'] = y_train.shape
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(os.path.join(save_path, 'config.json'), 'w') as outfile:
                json.dump(conf, outfile)

            ## define model on first epoch
            TCD = tempConvDecoder(**conf_nn)

        print('epoch:', epoch)
        TCD.model.fit(
            X_train,
            y_train,
            epochs=1, 
            batch_size=conf_nn['bs']
        )
        
        R2s,rs = TCD.determine_fit(X_test, y_test)
        stats.append([R2s, rs])
    
        print("R2: %s\n r: %s" % (R2s, rs))
        gc.collect()
        tr.print_diff()
    
    print(stats)
