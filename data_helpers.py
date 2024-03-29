from scipy import stats, signal
import numpy as np
from itertools import zip_longest

def format_timeseries(timeseries1, timeseries2, window, offset, shuffle_window=600, discard_buffer=50, 
                      standardize=True, percent_data=1.0, resample_data=False, sample_size=40000, regressor=True):
    X, y = make_timeseries_instances(timeseries1, timeseries2, window, offset)
    if resample_data:
        X, y = resample_Xy(X, y, sample_size=sample_size)
    if not regressor:
        X, y = thresh_and_label(X,y,threshold=0.2)
        X, y = timeseries_shuffler(X, y, shuffle_window, 1)
    else:
        X, y = timeseries_shuffler(X, y, shuffle_window, discard_buffer)
    X_train, X_test, y_train, y_test = split_data(X, y, 0.2, standardize)
    print('X_train and y_train shape:', X_train.shape, y_train.shape)
    if percent_data < 1.0:
        X_train, y_train = percent_train(X_train, percent_data), percent_train(y_train, percent_data)
    return X_train, X_test, y_train, y_test

def percent_train(train,percent_data):
    percent_idx = int(train.shape[0]*percent_data)
    return train[:percent_idx]

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

def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

def get_turn_peaks(dx, threshold):
    ## ephys = samples x electrode channels
    crossings =  np.where(abs(dx) > threshold)[0]
    peaks = []
    grouped_crossings = group_consecutives(crossings)
    for idx,thing in enumerate(grouped_crossings):
        center = thing[np.argmax(abs(dx[thing]))]
        peaks.append(center)
        
    return peaks

def thresh_and_label(X, y, threshold=.5):
    peaks = get_turn_peaks(y,threshold=threshold)

    X_peaks = X[peaks,:]

    labels = []
    for peak in peaks:
        if y[peak] > 0:
            labels.append(1)
        elif y[peak] < 0:
            labels.append(0)
    labels = np.atleast_2d(np.array(labels)).T
    return X_peaks, labels

def sample_inv_dist(y, sample_size=10000, bins=10000):
    ################### sample the dx distribution evenly: ###################
    y = np.squeeze(y)
    hist,edges = np.histogram(y,bins=bins,normed=True)
    bins_where_values_from = np.searchsorted(edges,y)
    bin_weights = 1/(hist/sum(hist))
    inv_weights = bin_weights[bins_where_values_from-1]
    dx_idx = np.arange(0,len(y),1)
    sampled_dx_idx = np.random.choice(dx_idx,size=sample_size,replace=False,p=inv_weights/sum(inv_weights))
#     sampled_dx = np.random.choice(y,size=sample_size,replace=False,p=inv_weights/sum(inv_weights))
    return np.sort(sampled_dx_idx)

def resample_Xy(X,y,sample_size):
    resampled_idxs = sample_inv_dist(y,sample_size=sample_size)
    return X[resampled_idxs], y[resampled_idxs]