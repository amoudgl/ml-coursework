import numpy as np
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32" 
np.random.seed(1337) # for reproducibility
#Load the dataset
from keras.datasets import mnist
# Import the sequential module from keras
from keras.models import Sequential
# Import the layers you wish to use in your net
from keras.layers.core import Dense, Dropout, Activation
# Import the optimization algorithms that you wish to use
from keras.optimizers import SGD, Adam, RMSprop
# Import other utilities that help in data formatting etc. 
from keras.utils import np_utils

batch_size = 512
nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
# We need to shape the data into a shape that network accepts.
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Here we convert the data type of data to 'float32'
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# We normalize the data to lie in the range 0 to 1.
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Dense(10))
model.add(Activation('softmax'))
model1 = Sequential() #change batch size 
model1.add(Dense(512, input_shape=(784,))) 
model1.add(Activation('relu')) 
model1.add(Dense(512)) 
model1.add(Activation('relu')) 
model1.add(Dense(10)) 
model1.add(Activation('softmax'))
model2 = Sequential() #change hidden layers 
model2.add(Dense(1024, input_shape=(784,))) 
model2.add(Activation('relu')) 
model2.add(Dense(1024)) 
model2.add(Activation('relu'))
model2.add(Dense(10)) 
model2.add(Activation('softmax'))
model3 = Sequential() #change activation function 
model3.add(Dense(512, input_shape=(784,))) 
model3.add(Activation('sigmoid'))
model3.add(Dense(512)) 
model3.add(Activation('sigmoid')) 
model3.add(Dense(10)) 
model3.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
model1.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
model2.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
model3.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

history = model.fit(X_train, Y_train,batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
history1 = model.fit(X_train, Y_train,batch_size=1024, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
history2 = model.fit(X_train, Y_train,batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
history3 = model.fit(X_train, Y_train,batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

import matplotlib.pyplot as plt 
         # list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show() #-------------------------------------------------------------------- 
plt.plot(history1.history['acc']) 
plt.plot(history1.history['val_acc'])
plt.title('model 1 accuracy - Batch size (1024)') 
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show() #------------------------------------------------------------------- 
plt.plot(history2.history['acc']) 
plt.plot(history2.history['val_acc'])
plt.title('model 2 accuracy - Hidden layers (1024)') 
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show() #------------------------------------------------------------------- 
plt.plot(history3.history['acc']) 
plt.plot(history3.history['val_acc'])
plt.title('model 3 accuracy - Activation function (sigmoid)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show() #-------------------------------------------------------------------
# summarize history for loss
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show() #------------------------------------------------------------------ 
plt.plot(history1.history['loss']) 
plt.plot(history1.history['val_loss'])
plt.title('model 1 loss - Batch size (1024)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show() #------------------------------------------------------------------ 
plt.plot(history2.history['loss']) 
plt.plot(history2.history['val_loss'])
plt.title('model 2 loss - Hidden layers (1024)') 
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show() #------------------------------------------------------------------ 
plt.plot(history3.history['loss']) 
plt.plot(history3.history['val_loss'])
plt.title('model 3 loss - Activation function (sigmoid)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


