import numpy as np
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
        self.verbose = kwargs['verbose']
        self.key = kwargs['key']
        self.nb_input_samples, _, self.nb_input_series = kwargs['dataset']['train'][0].shape
        self.nb_output_samples, self.nb_output_series = kwargs['dataset']['train'][1].shape
        self.X_train, self.y_train = kwargs['dataset']['train'] 
        self.X_test, self.y_test = kwargs['dataset']['test']

    def make_timeseries_regressor(self):
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

    def fit(self):
        model = self.make_timeseries_regressor()
        if self.verbose:
            print('\n\nTimeseries ({} samples by {} series):\n'.format(self.nb_input_samples, self.nb_input_series))
            print('\n\nExample input feature:', X[0], '\n\nExample output labels:', y[0])
            print('\n\nModel with input size {}, output size {}'.format(
                model.input_shape,
                model.output_shape
            ))
            model.summary()

        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=2,
            verbose=0,
            mode='auto'
        )

        model.fit(
            self.X_train,
            self.y_train,
            epochs=self.eps,
            batch_size=self.bs,
            validation_data=(self.X_test, self.y_test),
            callbacks=[early_stopping]
        )

        self.model = model

    def determine_fit(self, plot_result=False):
        self.y_test_hat = self.model.predict(self.X_test)

        R2s, rs = do_the_thing(
            self.y_test,
            self.y_test_hat,
            self.key,
            'temp_conv_results_{}_y:{}'.format(self.run_id, self.key),
            plot_result=plot_result
        )

        return R2s, rs
