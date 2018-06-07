import os
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from metrics_helper import do_the_thing

class tempConvDecoder(object):
    """A class that stores the network, trains it, and determines it's goodness of fit."""

    def __init__(self, **kwargs):
        self.bs  = kwargs['bs']
        self.eps = kwargs['eps']
        self.lr  = kwargs['lr']
        self.kernel = kwargs['kernel']
        self.nb_filter = kwargs['nb_filter']
        self.window = kwargs['window']
        self.regressor = kwargs['regressor']
        self.pyramidal = kwargs['pyramidal']
        self.run_id = kwargs['run_id']
        self.save_path = kwargs['save_path']
        self.verbose = kwargs['verbose']
        self.key = kwargs['key']
        self.nb_input_samples, _, self.nb_input_series = kwargs['input_shape']
        self.nb_output_samples, self.nb_output_series = kwargs['output_shape']
        self.model_type = kwargs['model_type']

        if self.model_type == 'ridge':
            model = self.make_ridgeCV_model()
        elif self.model_type == 'conv':
            model = self.make_timeseries_regressor()

        self.model = model
    
    def make_ridgeCV_model():
         
        print('********************************** Making RidgeCV Model **********************************')
        #Declare model
        model = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0],normalize=True,fit_intercept=True)
                    
        return model

    def make_timeseries_regressor(self):
        print('********************************** Making 1D Conv Model **********************************')
        dilation = [4,2] if self.pyramidal else [1,1]
        model = Sequential()
        model.add(Conv1D(
            int(self.nb_filter * dilation[0]),
            kernel_size=int(self.kernel * dilation[0]),
            activation='relu',
            input_shape=(self.window, self.nb_input_series)
        ))
        model.add(MaxPooling1D())
        ## dupes >
        model.add(Conv1D(
            int(self.nb_filter * dilation[1]),
            kernel_size=int(self.kernel * dilation[1]),
            activation='relu'
        ))
        model.add(MaxPooling1D())
        ## < dupes
        model.add(Conv1D(
            int(self.nb_filter * dilation[1]),
            kernel_size=int(self.kernel * dilation[1]),
            activation='relu'
        ))
        model.add(MaxPooling1D())
        model.add(Conv1D(
            int(self.nb_filter),
            kernel_size=int(self.kernel),
            activation='relu'
        ))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(self.nb_filter * 12, activation='relu'))
        model.add(Dropout(0.5))
        ## dupes >
        model.add(Dense(self.nb_filter * 12, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_filter * 8, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_filter * 4, activation='relu'))
        model.add(Dropout(0.5))
        ## < dupes
        adam = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        
        if self.regressor:
            model.add(Dense(self.nb_output_series, activation='linear'))
            model.compile(loss='mae', optimizer=adam, metrics=['mse'])
        else:
            model.add(Dense(self.nb_output_series, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
 
        return model

    def determine_fit(self, X_test=None, y_test=None, save_result=True):
        if X_test == None:
            X_test = self.X_test
            y_test = self.y_test
        y_test_hat = self.model.predict(X_test)

        R2s, rs = do_the_thing(
            y_test,
            y_test_hat,
            self.key,
            '{}_results_{}_y:{}'.format(self.model_type, self.run_id, self.key),
            os.path.join(self.save_path,self.run_id),
            save_result=save_result
        )

        return R2s, rs
