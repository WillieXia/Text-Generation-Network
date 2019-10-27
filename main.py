import numpy
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#Setting up the text
huckleberry = "huckleberry.txt"
huckle_raw = open(huckleberry, 'r', encoding='utf-8').read()
huckle_raw = huckle_raw.lower()

#Setting up letters to integers
char_map = sorted(list(set(huckle_raw)))
char_to_int = dict((c, i) for i, c in enumerate(char_map))

n_vocab = len(char_map)

breaksection = 100
Xdata = []
Ydata= []
for i in range(0, len(huckle_raw)-breaksection, 1):
    seq_in = huckle_raw[i:i + breaksection]
    seq_out = huckle_raw[i+breaksection]
    Xdata.append([ord(letter) for letter in seq_in])
    Ydata.append(char_to_int[seq_out])
  
#Reshaping and normalizing
X = numpy.reshape(Xdata, (len(Xdata), breaksection, 1))
X = X / float(len(char_map))
Y = np_utils.to_categorical(Ydata)

model = keras.models.Sequential()
model.add(keras.layers.LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

#Saving checkpoint Models
filepath="Huckleberry-Model-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#Fitting the model/#Tests
model.fit(X, Y, epochs=20, batch_size=128, callbacks=callbacks_list)