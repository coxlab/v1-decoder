from scipy import stats, signal
import numpy as np
from itertools import zip_longest

def format_timeseries(timeseries1, timeseries2, window, offset):
    X, y = make_timeseries_instances(timeseries1, timeseries2, window, offset)
    X, y = timeseries_shuffler(X, y, 3000, 25)
    return split_data(X, y, 0.5)

def make_timeseries_instances(X, y, window_size, offset):
    X = np.asarray(X)
    y = np.asarray(y)
    assert 0 < window_size < X.shape[0]
    assert X.shape[0] == y.shape[0]
    X = np.atleast_3d(np.array([X[start:start+window_size] for start in range(0, X.shape[0] - window_size)]))
    y = y[window_size:]
    print('pre-offset',len(X))
    if offset > 0:
        X,y = X[:-offset], y[offset:]
    elif offset < 0:
        X,y = X[-offset:],y[:offset]
    print('post-offset',len(X))
    return X, y

def split_data(X, y, test_size, standardize=True):
    test_size_idx = int(test_size * X.shape[0])
    X_train, X_test, y_train, y_test = X[:-test_size_idx], X[-test_size_idx:], y[:-test_size_idx], y[-test_size_idx:]
    if standardize:
        #Z-score "X" inputs. 
        X_train_mean = np.nanmean(X_train, axis=0)
        X_train_std = np.nanstd(X_train, axis=0)
        X_train = (X_train - X_train_mean) / X_train_std
        X_test = (X_test - X_train_mean) / X_train_std

        #Zero-center outputs
        #y_train_mean = np.mean(y_train, axis=0)
        #y_train = y_train - y_train_mean
        #y_test = y_test - y_train_mean
    
    return X_train, X_test, y_train, y_test

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