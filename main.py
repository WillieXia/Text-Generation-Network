import numpy
import tensorflow as tf
import keras
import sys
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

#Setting up the text
huckleberry = "Dark_Knight_Script.txt"
huckle_raw = open(huckleberry, 'r', encoding='utf-8').read() #Reading each line and lower casing them
huckle_raw = huckle_raw.lower()

#Setting up letters to integers
char_map = sorted(list(set(huckle_raw))) 
int_to_char = dict((i, c) for i, c in enumerate(char_map)) #Mapping each character to a number, and number to character
char_to_int = dict((c, i) for i, c in enumerate(char_map))

n_vocab = len(char_map) #Help to see when to break
n_chars = len(huckle_raw)

breaksection = 100
Xdata = []
Ydata= []
for i in range(0, n_chars-breaksection, 1): #Inputting each character into our X layer and Y layer
    seq_in = huckle_raw[i:i + breaksection]
    seq_out = huckle_raw[i+breaksection]
    Xdata.append([char_to_int[char] for char in seq_in])
    Ydata.append(char_to_int[seq_out])
 
#Reshaping and normalizing
Xp = numpy.reshape(Xdata, (len(Xdata), breaksection, 1)) #This helps us form the data to be readable by our network
Xp = Xp / float(n_vocab)
Yp = np_utils.to_categorical(Ydata)

model = keras.models.Sequential() #Creating the model and adding in new layers.
model.add(keras.layers.LSTM(256, input_shape=(Xp.shape[1], Xp.shape[2])))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(Yp.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

#filepath="batman-model-{epoch:02d}-{loss:.4f}.hdf5"
#checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
#callbacks_list = [checkpoint]
#model.fit(Xp, Yp, epochs=20, batch_size=128, callbacks=callbacks_list)

filename = "batman-model-20-2.0439.hdf5"
model.load_weights(filename)



###saving checkpoint models
#filepath="huckleberry-model-{epoch:02d}-{loss:.4f}.hdf5"
#checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
#callbacks_list = [checkpoint]

##fitting the model/#tests
#model.fit(Xp, Yp, epochs=20, batch_size=128, callbacks=callbacks_list)

# pick a random seed
start = numpy.random.randint(0, len(Xdata)-1)
pattern = Xdata[start]
print ("Seed:")
concat = "/"
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
for i in range(1000): #Print 1000 characters at a time.
    xout = numpy.reshape(pattern, (1, len(pattern), 1))
    xout = xout / float(n_vocab)
    prediction = model.predict(xout, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print ("\nDone.")
