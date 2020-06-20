#Infrastructure
from sklearn.utils.validation import NotFittedError

#NN
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


#Data Handling
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import NotFittedError

#Utils
import numpy as np
from sklearn.preprocessing import StandardScaler
#for voter
from sklearn.neighbors import KNeighborsRegressor

class NNVoter(object):
    def __init__(
        self,
        X, y,
        validation_split = 0.25,
        input_dim = 4,
        epochs = 100,
        lr = 1e-4,
        verbose = False,
    ):
        self.voter = keras.Sequential()
        self.voter.add(layers.Dropout(0.2, input_shape=(input_dim,)))
        self.voter.add(layers.Dense(1, activation='linear', input_shape=(input_dim,) ))
        self.voter.compile(loss = 'mse', metrics=['mae'], optimizer = keras.optimizers.Adam(lr))           
        self.voter.fit(X, 
                    y, 
                    epochs = epochs, 
                    callbacks = [EarlyStopping(patience = 20, monitor = "val_loss")], 
                    verbose = verbose, validation_split = validation_split,
                    shuffle=True, )
                
    def predict(self, X):
        return self.voter.predict(X)
    

class HonestRegression():
    def __init__(
        self,
        calibration_split = .3,
        encoder_dim = 4,
        verbose = False,
        auto_encoder_transformer =True,
    ):
        self.calibration_split = calibration_split
        self.transformer_fitted_ = False
        self.voter_fitted_ = False
        self.encoder_dim = encoder_dim
        self.verbose = verbose
        self.scaler_x = None
        self.scaler_y = None
        self.auto_encoder_transformer = auto_encoder_transformer
    def check_transformer_fit_(self):
        '''
        raise a NotFittedError if the transformer isn't fits
        '''
        if not self.transformer_fitted_:
                msg = (
                        "This %(name)s instance's transformer is not fitted yet. "
                        "Call 'fit_transform' or 'fit' with appropriate arguments "
                        "before using this estimator."
                )
                raise NotFittedError(msg % {"name": type(self).__name__})
                
    def check_voter_fit_(self):
        '''
        raise a NotFittedError if the voter isn't fit
        '''
        if not self.voter_fitted_:
                msg = (
                        "This %(name)s instance's voter is not fitted yet. "
                        "Call 'fit_voter' or 'fit' with appropriate arguments "
                        "before using this estimator."
                )
                raise NotFittedError(msg % {"name": type(self).__name__})
    
    def check_scalers_fit_(self):
        '''
        raise a NotFittedError if the voter isn't fit
        '''
        if  self.scaler_x is None:
                msg = (
                        "This %(name)s instance's x_scaler is not fitted yet. "
                        "Call  'fit' with appropriate arguments "
                        "before using scaler."
                )
                raise NotFittedError(msg % {"name": type(self).__name__})
                
        if  self.scaler_y is None:
                msg = (
                        "This %(name)s instance's y_scaler is not fitted yet. "
                        "Call  'fit' with appropriate arguments "
                        "before using scaler."
                )
                raise NotFittedError(msg % {"name": type(self).__name__})
    
    def fit_transformer(self, X, y, epochs = 100, lr = 5e-5):            
        input_dim = X.shape[1]
        if self.auto_encoder_transformer:
            self.network = keras.Sequential()
            self.network.add(layers.Dense(256, activation='relu', input_shape=(input_dim,)))
            self.network.add(layers.Dropout(0.2))
            self.network.add(layers.Dense(self.encoder_dim, activation='relu'))
            self.network.add(layers.Dense(256, activation='relu', input_shape=(input_dim,)))
            self.network.add(layers.Dense(input_dim, activation='linear'))
            self.network.compile(loss = 'mse', metrics=['mae'], optimizer = keras.optimizers.Adam(lr))
            self.network.fit(
                        X, 
                        X, 
                        epochs = epochs, 
                        callbacks = [EarlyStopping(patience = 10, monitor = "val_loss")], 
                        verbose = self.verbose,
                        validation_split = .25,
                        shuffle=True, )

            self.encoder = keras.models.Model(inputs = self.network.inputs, outputs = self.network.layers[-3].output)
        else:
            self.network = keras.Sequential()
            self.network.add(layers.Dense(256, activation='relu', input_shape=(input_dim,)))
            self.network.add(layers.Dense(self.encoder_dim, activation='relu'))
            self.network.add(layers.Dropout(0.2))
            self.network.add(layers.Dense(1, activation='linear'))
            self.network.compile(loss = 'mse', metrics=['mae'], optimizer = keras.optimizers.Adam(lr))
            self.network.fit(X, 
                     y, 
                     epochs = epochs, 
                     callbacks = [EarlyStopping(patience = 10, monitor = "val_loss")], 
                     verbose = self.verbose,
                     validation_split = .25,
                     shuffle=True, )

            self.encoder = keras.models.Model(inputs = self.network.inputs, outputs = self.network.layers[-3].output)

        #make sure to flag that we're fit
        self.transformer_fitted_ = True
    
    def fit_knn_voter(self, X, y):        
        self.knn = KNeighborsRegressor(16 * int(np.log2(len(X))), weights = "distance", p = 1)
        self.knn.fit(X, y)
        
        #make sure to flag that we're fit
        self.voter_fitted_ = True
    
    def fit_voter(self, X, y, epochs=100, lr=1e-4):
        self.voter = NNVoter(X, y,  
        validation_split = 0.25,
        input_dim = self.encoder_dim,
        verbose = self.verbose,
        epochs = epochs,
        lr = lr)
        
        #make sure to flag that we're fit
        self.voter_fitted_ = True

    def fit(self, X, y, epochs = 100, lr = 1e-4):
        
        self.scaler_x = StandardScaler().fit(X)
        self.scaler_y = StandardScaler().fit(y.reshape(-1,1))
        
        X_trans = self.scaler_x.transform(X)
        y_trans = self.scaler_y.transform(y.reshape(-1,1))
        
        #split
        X_train, X_cal, y_train, y_cal = train_test_split(X_trans, y_trans, test_size = self.calibration_split)
        
        #fit the transformer
        self.fit_transformer(X_train, y_train, epochs = epochs, lr = lr)

        #fit the voter
        X_cal_transformed = self.transform(X_cal, do_scale=False)
        self.fit_voter(X_cal_transformed, y_cal,  epochs=epochs, lr = lr )
        self.fit_knn_voter(X_cal_transformed, y_cal)
        
    def transform(self, X, do_scale=True):
        self.check_transformer_fit_()
        if do_scale:
            return self.encoder.predict(self.scaler_x.transform(X))
        else:
            return self.encoder.predict(X)

    def vote(self, X_transformed):
        self.check_voter_fit_()
        return self.voter.predict(X_transformed)
    
    def get_voter(self):
        self.check_voter_fit_()
        return self.voter
    
    def predict(self, X):
        return self.scaler_y.inverse_transform(self.vote(self.transform(X)))
    
    def scale_X(self, X):
        self.check_scalers_fit_()
        return self.scaler_x.transform(X, copy=True)
    
    def scale_y(self, y):
        self.check_scalers_fit_()
        return self.scaler_y.transform(y.reshape(-1,1))
    
    def inverse_scale_y(self, y):
        self.check_scalers_fit_()
        return self.scaler_y.inverse_transform(y.reshape(-1,1))
    
    
    