from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding,Dropout
from keras.preprocessing import sequence
from keras.utils import to_categorical
import scipy.io as sio
from keras import backend as K
import h5py
import numpy as np
import os
import keras
from numpy import array
np.random.seed(1337)  # for reproducibility
from keras.models import Model
from keras.layers import Input
from keras import layers, models
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects

#swish define
def custom_activation(x):
    return (K.sigmoid(x) * x)
get_custom_objects().update({'custom_activation': Activation(custom_activation)})

### data loading
# def load_data():
#     with h5py.File('./depression_dataset.h5')as hf:
#         x_train=hf['train'][:].transpose()
#         y_train=hf['train_label'][:].transpose()
#         x_test=hf['test'][:].transpose()
#         y_test=hf['test_label'][:].transpose()
#         return (x_train, y_train), (x_test, y_test)

def build_model_LSTM():
    inputs = Input(shape=(527,1,))
    x = LSTM(512,return_sequences=True)(inputs)
    x = LSTM(256,return_sequences=True)(x)
    x = LSTM(64,return_sequences=False)(x)
    output_dep = Dense(1, activation='sigmoid')(x)
    model = Model(input=[inputs], output=[output_dep])
    model.summary()
    return model

def build_model_DNN():
    inputs = Input(shape=(527,))
    x = Dense(512,activation='relu')(inputs)
    x = Dense(1024,activation='relu')(x)
    x = Dense(4096,activation='relu')(x)
    output_dep = Dense(1, activation='sigmoid')(x)
    model = Model(input=[inputs], output=[output_dep])
    model.summary()
    return model

# load data
# (x_train, y_train), (x_test, y_test) = load_data()

model = build_model_DNN()
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
checkpoint = keras.callbacks.ModelCheckpoint('./models/best_weights.h5', monitor='val_dense_4_loss', save_best_only=True,                                            verbose=1,mode='min')
early_stopping=keras.callbacks.EarlyStopping(monitor='val_dense_4_loss', patience=20, verbose=1, mode='min')
model.compile(
    optimizer=sgd,
    loss='binary_crossentropy', #similar to tf sigmoid_cross_entropy_with_logits
    metrics=['acc']
)
history = model.fit(x_train, y_train, epochs=100, verbose=0)
score=model.evaluate(x_test, y_test, verbose=1)