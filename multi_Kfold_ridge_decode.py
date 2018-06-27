
# coding: utf-8

# In[9]:

import os
import gc
import sys
import numpy as np
import json
import re
from pympler import tracker
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from data_helpers import sample_train_test, load_data
from metrics_helper import do_the_thing
from other import get_sha
tr = tracker.SummaryTracker()

conf_path = os.path.abspath(sys.argv[1]) # pass path to config here as arg
rat_path = os.path.dirname(conf_path)
conf = json.load(open(conf_path, 'r'))
conf['git_version'] = get_sha()

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
        if True in [conf['config']['neural_data'] in fil for fil in data_file_list]:
            dataset_paths.append([
                folder+'/'+data_file_list[data_file_list.index(X_fname)],
                folder+'/'+data_file_list[data_file_list.index(y_fname)]
            ])

dataset_paths = np.array(dataset_paths)

# collect statistics from each epoch
full_results = {
    'stats': []
}

def make_ridgeCV_model():    
    print('********************************** Making RidgeCV Model **********************************')
    #Declare model
    model = Ridge(alpha=0.1, normalize=True,fit_intercept=True)
  
    return model

def run_Kfold(train_idx, test_idx):
    train_paths = dataset_paths[train_idx]
    test_paths = dataset_paths[test_idx]
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
    
    results = {}
    exp_num_regex = re.compile('\d{18}')

    ## train for number of epochs, resampling training datasets at each new epoch
    #### train model and test model for each Kfold
    # sample first round of data:
    X_train, y_train = load_data(train_paths, conf_nn, sample_frac=conf_nn['X_frac'])
    print("X_train.shape:", X_train.shape)
    X_train = X_train.reshape(X_train.shape[0],(X_train.shape[1]*X_train.shape[2]))

    model = make_ridgeCV_model()
    model.fit(X_train, y_train)
    X_train, y_train = [None, None] # delete residual data to free up space
    
    for test_path in test_paths:
        exp_num = re.search(exp_num_regex, test_path[0]).group()
        results[exp_num] = []
        print('indie test on: ', test_path)
        X_test, y_test = load_data([test_path], conf_nn, sample_frac=1)
        X_test = X_test.reshape(X_test.shape[0],(X_test.shape[1]*X_test.shape[2]))
        # ugly plot results if final epoch
        y_test_hat = model.predict(X_test)
        R2s, rs = do_the_thing(
            y_test,
            y_test_hat,
            conf_nn['key'],
            '{}_results_y:{}'.format(conf_nn['model_type'], conf_nn['key']),
            conf_nn['save_path'],
            save_result=True
        )
            
        results[exp_num].append([R2s,rs])

    print("R2: %s\n r: %s" % (R2s, rs))

    gc.collect()
    tr.print_diff()

    conf_nn['results'] = results
    full_results['stats'].append(results)
    # save stats and settings after last epoch
    with open(os.path.join(save_path, 'trial_info.json'), 'w') as outfile:
        json.dump(conf, outfile)
    
    print(results)

for i in range(1, int(len(dataset_paths)/2)+1):
    kf = KFold(n_splits=int(len(dataset_paths)/i))
    for train_idx, test_idx in kf.split(dataset_paths):
    	run_Kfold(train_idx, test_idx)
    	run_Kfold(test_idx, train_idx)

with open(os.path.join(rat_path, conf['nn_params']['model_type']+'_results.json'), 'w') as outfile:
    json.dump(full_results, outfile)
