# compare different loss functions on MNIST dataset

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

batchSize = 512                    #-- Training Batch Size
num_classes = 10                  #-- Number of classes in CIFAR-10 dataset
num_epochs = 20                   #-- Number of epochs for training   
learningRate= 0.001               #-- Learning rate for the network
lr_weight_decay = 0.95            #-- Learning weight decay. Reduce the learn rate by 0.95 after epoch

img_rows, img_cols = 28, 28  

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

for i in xrange(4):
    model = Sequential()                                                
    #-- layer 1
    model.add(Convolution2D(6, 5, 5,                                    
                           border_mode='valid',
                           input_shape=(1, img_rows, img_cols) ,dim_ordering="th")) 
    model.add(Convolution2D(16, 5, 5, dim_ordering="th"))
    model.add(Activation('relu'))                                       
    model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
    #--layer 2
    model.add(Convolution2D(26, 3, 3,dim_ordering="th"))
    model.add(Convolution2D(36, 3, 3,dim_ordering="th"))
    model.add(Activation('relu'))                                       
    model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
    #-- layer 4
    model.add(Flatten())                                                
    model.add(Dense(256))                                               
    model.add(Activation('relu'))                                       
    #-- layer 5
    model.add(Dense(256))                                                
    model.add(Activation('relu'))                                       
    #-- layer 6
    model.add(Dense(10))                                       
    #-- loss
    if (i > 1):
        model.add(Activation('softmax')) #-- converts the output to a log-probability. Useful for classification problems
    print(model.summary())
    models.append(model)


def hinge_onehot(y_true, y_pred):
    y_true = y_true*2 - 1
    y_pred = y_pred*2 - 1
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)

def hinge_square_onehot(y_true, y_pred):
    y_true = y_true*2 - 1
    y_pred = y_pred*2 - 1
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)

losses = []

losses.append(hinge_onehot)
losses.append(hinge_square_onehot)
losses.append('categorical_crossentropy')
losses.append('kullback_leibler_divergence')

histories = []

for model, loss in zip(models, losses):
    sgd = SGD(lr=learningRate, decay = lr_weight_decay)
    model.compile(loss=loss,
                  optimizer='sgd',
                  metrics=['accuracy'])
    #-- switch verbose=0 if you get error "I/O operation from closed file"
    history = model.fit(X_train, Y_train, batch_size=batchSize, nb_epoch=num_epochs,
              verbose=1, shuffle=True, validation_data=(X_test, Y_test))
    histories.append(history)
    
for history in histories:
    plt.plot(history.history['val_acc'])

plt.title('Model\' Validation Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Hinge', 'SquaredHinge', 'CrossEntropy', 'KLD'], loc='lower right')
plt.show()

for history in histories:
    plt.plot(history.history['loss'])

plt.title('Model\' Training Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Hinge', 'SquaredHinge', 'CrossEntropy', 'KLD'], loc='upper right')
plt.show()

