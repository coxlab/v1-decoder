import numpy as np
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from data_helpers import format_timeseries
from metrics_helper import do_the_thing

class tempConvDecoder(object):
    """A class that stores the network, trains it, and determines it's goodness of fit."""

    def __init__(self, timeseries1, timeseries2, key, bs=256, eps=15, lr=0.0005, kernel=2, 
                 nb_filter=5, window=60, offset=10, id=None, nb_trains=1, verbose=False):
        self.bs = bs
        self.eps = eps
        self.lr = lr
        self.kernel = kernel
        self.nb_filter = nb_filter
        self.window = window
        self.offset = offset
        self.id = id
        self.nb_trains = nb_trains
        self.verbose = verbose
        self.key = key
        self.timeseries1 = timeseries1
        self.nb_input_samples, self.nb_input_series = timeseries1.shape
        if timeseries2.ndim == 1: timeseries2 = np.atleast_2d(timeseries2).T
        self.timeseries2 = timeseries2
        self.nb_output_samples, self.nb_output_series = timeseries2.shape
        self.X_train, self.X_test, self.y_train, self.y_test = format_timeseries(
            timeseries1,
            timeseries2, 
            window, 
            offset
        )

    def make_timeseries_regressor(self):
        model = Sequential()
        model.add(Conv1D(
            int(self.nb_filter * 4),
            kernel_size=int(self.kernel * 4),
            activation='relu',
            input_shape=(self.window, self.nb_input_series)
        ))
        model.add(MaxPooling1D())
        model.add(Conv1D(
            int(self.nb_filter * 2),
            kernel_size=int(self.kernel * 2),
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
        model.add(Dense(self.nb_filter * 8, activation='relu'))
        model.add(Dropout(0.5))
        # output layer: for binary classification, change the activation to 'sigmoid'
        model.add(Dense(self.nb_output_series, activation='linear')) 
        adam = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='mae', optimizer=adam, metrics=['mse'])
        # To perform (binary) classification instead:
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
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
            patience=1,
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
            'temp_conv_results_{}_y:{}'.format(self.id, self.key),
            plot_result=plot_result
        )

        return R2s, rs