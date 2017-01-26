# compare two models with different number of layers on MNIST dataset

from __future__ import print_function
import numpy as np
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

from keras import backend as K

#Load the dataset
from keras.datasets import mnist

from keras.models  import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt

batchSize = 128                   #-- Training Batch Size
num_classes = 10                  #-- Number of classes in CIFAR-10 dataset
num_epochs = 10                   #-- Number of epochs for training   
learningRate= 0.001               #-- Learning rate for the network
lr_weight_decay = 0.95            #-- Learning weight decay. Reduce the learn rate by 0.95 after epoch

img_rows, img_cols = 28, 28  
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

"""Load and format Data
"""
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
# We need to shape the data into a shape that network accepts.
X_train = X_train.reshape(60000, 1, 28, 28)
X_test = X_test.reshape(10000, 1, 28, 28)

# Here we convert the data type of data to 'float32'
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# We normalize the data to lie in the range 0 to 1.
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

models = []

model1 = Sequential()                                                
#-- layer 1
model1.add(Convolution2D(6, 5, 5,                                    
                       border_mode='valid',
                       input_shape=(1, img_rows, img_cols) ,dim_ordering="th")) 
model1.add(Convolution2D(16, 5, 5, dim_ordering="th"))
model1.add(Activation('relu'))                                       
model1.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
#--layer 2
model1.add(Convolution2D(26, 3, 3,dim_ordering="th"))
model1.add(Convolution2D(36, 3, 3,dim_ordering="th"))
model1.add(Activation('relu'))                                       
model1.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
#--layer 3
model1.add(Flatten())                                                
model1.add(Dense(256))                                               
model1.add(Activation('relu'))                                       
#-- layer 4
model1.add(Dense(256))                                                
model1.add(Activation('relu'))                                       
#-- layer 5
model1.add(Dense(num_classes))                                       
#-- loss
model1.add(Activation('softmax'))                                    #-- converts the output to a log-probability. Useful for classification problems
print(model1.summary())
models.append(model1)


model2 = Sequential()
model2.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model2.add(Activation('relu'))
model2.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model2.add(Dropout(0.25))

model2.add(Flatten())
model2.add(Dense(128))
model2.add(Activation('relu'))
model2.add(Dropout(0.5))
model2.add(Dense(num_classes))
model2.add(Activation('softmax'))
models.append(model2)

histories = []

for model in models:
    sgd = SGD(lr=learningRate, decay = lr_weight_decay)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    #-- switch verbose=0 if you get error "I/O operation from closed file"
    history = model.fit(X_train, Y_train, batch_size=batchSize, nb_epoch=num_epochs,
              verbose=1, shuffle=True, validation_data=(X_test, Y_test))
    histories.append(history)

for history in histories:
    plt.plot(history.history['val_acc'])

plt.title('Models\' Validation Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Deep Net', 'Default Net'], loc='lower right')
plt.show()

for history in histories:
    plt.plot(history.history['loss'])

plt.title('Models\' Training Loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Deep Net', 'Default Net'], loc='upper right')
plt.show()

