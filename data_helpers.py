import os
import random
import numpy as np
import h5py
from sklearn import preprocessing
from itertools import zip_longest

def load_data(paths, conf, sample_frac=1):
    X, y = format_timeseries(
        paths, 
        conf['window'],
        conf['offset'],
        conf['X_split'], 
        conf['y_split'],
        sample_frac=sample_frac
    )
    print('Loaded data from %s' % paths)
    
    return X, y

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
    print('sample test loaded: %s' % test_paths)
    
    return X_train, y_train, X_test, y_test

def format_timeseries(dataset_paths, window, offset, xnds, ynds, 
                       shuffle_window=600, discard_buffer=50, sample_frac=1):
    X, y, = [], []
    for X_i, y_i in dataset_paths:
        X_i, y_i = read_data_pair(X_i, y_i)
        X_i, y_i = X_i[:, slice(*xnds)], y_i[:, slice(*ynds)]
        X_i, y_i = preprocessing.scale(X_i), y_i
        X_i, y_i = make_timeseries_instances(X_i, y_i, window, offset)
        ## sub sample from timeseries frames if sample frac
        if not sample_frac == 1:
            r_idxs = random.sample(range(len(list(X_i))),int(len(X_i)*sample_frac))
            X_i, y_i = X_i[r_idxs], y_i[r_idxs]
#       X_i, y_i = timeseries_shuffler(X_i, y_i, shuffle_window, discard_buffer)
        X.append(X_i)
        y.append(y_i)
    X = np.concatenate(X)
    y = np.concatenate(y)
    
    return X, y

def ts_chnls(data):
    ## puts longest dimension (timeseries) first
    if data.shape[0] < data.shape[1]:
        data = data.T
    return data

def read_data_pair(X_path, y_path):
    X = h5py.File(X_path, 'r')
    X = np.array([np.array(X[key]) for key in X.keys()])[0].T
    X = shape_data(X)
    
    y = h5py.File(y_path, 'r')
    y = np.array([np.array(y[key]) for key in y.keys()]).T
    y = shape_data(y)
    
    return X, y

def shape_data(data):
    if data.ndim == 1: data = np.atleast_2d(data)
    data = ts_chnls(data)
    return data

def make_timeseries_instances(X, y, window_size, offset):
    X = np.asarray(X)
    y = np.asarray(y)
    assert 0 < window_size < X.shape[0]
    assert X.shape[0] == y.shape[0]
    X = np.atleast_3d(np.array([X[start:start+window_size] for start in range(0, X.shape[0] - window_size)]))
    y = y[window_size:]
    #print('pre-offset',len(X))
    if offset > 0:
        X,y = X[:-offset], y[offset:]
    elif offset < 0:
        X,y = X[-offset:],y[:offset]
    #print('post-offset',len(X))
    return X, y

def split_data(X, y, test_size):
    test_size_idx = int(test_size * X.shape[0])
    X_train, X_test, y_train, y_test = X[:-test_size_idx], X[-test_size_idx:], y[:-test_size_idx], y[-test_size_idx:]
    
    return X_train, X_test, y_train, y_test

def standardize(X):
    #Z-score "X" inputs. 
    X_mean = np.nanmean(X, axis=0)
    X_std = np.nanstd(X, axis=0)
    X = (X - X_mean) / X_std
    return X

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return np.asarray(list(zip_longest(*args, fillvalue=fillvalue)))

def timeseries_shuffler(X, y, series_length, padding):
    """shuffle time series data, chunk by chunk into even and odd bins, discarding a pad
    between each bin.

    Keyword arguments:
    X -- input dataset
    y -- output dataset
    series_length -- length of chunks to bin by
    padding -- pad to discard between each chunk.
    """
    X_even = []
    X_odd = []
    y_even = []
    y_odd = []
    
    # state variable control which bin to place data into
    odd = False
    
    for i in range(X.shape[0]):
        # after series_length + padding, switch odd to !odd
        if (i%(series_length+padding)) == 0:
            odd = not odd

        # only add to bin during the series period, not the padding period
        if (i%(series_length+padding))<series_length:
            
            # put X[i] and y[i] into even/odd bins
            if odd:
                X_odd.append(X[i])
                y_odd.append(y[i])
            else:
                X_even.append(X[i])
                y_even.append(y[i])
    # concatenate back together
    X_even.extend(X_odd)
    y_even.extend(y_odd)
    # put them back into np.arrays
    X_shuffled = np.asarray(X_even)
    y_shuffled = np.asarray(y_even)
    
    return X_shuffled, y_shuffled
