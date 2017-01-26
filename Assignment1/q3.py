import numpy as np
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import theano
import keras

from keras.datasets import cifar10
from keras.models  import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt
#%matplotlib inline

# load cifar dataset
cifar10 = np.load('../../../data/lab2/lab2_data/cifar10_data.npz')
X_train = cifar10['X_train']
y_train = cifar10['y_train']
X_test = cifar10['X_test']
y_test = cifar10['y_test']

print "Training data:"
print "Number of examples: ", X_train.shape[0]
print "Number of channels:",X_train.shape[1] 
print "Image size:", X_train.shape[2], X_train.shape[3]
print
print "Test data:"
print "Number of examples:", X_test.shape[0]
print "Number of channels:", X_test.shape[1]
print "Image size:",X_test.shape[2], X_test.shape[3] 


# normalise data
print "mean before normalization:", np.mean(X_train) 
print "std before normalization:", np.std(X_train)

mean=[0,0,0]
std=[0,0,0]
newX_train = np.ones(X_train.shape)
newX_test = np.ones(X_test.shape)
for i in xrange(3):
    mean[i] = np.mean(X_train[:,i,:,:])
    std[i] = np.std(X_train[:,i,:,:])
    
for i in xrange(3):
    newX_train[:,i,:,:] = X_train[:,i,:,:] - mean[i]
    newX_train[:,i,:,:] = newX_train[:,i,:,:] / std[i]
    newX_test[:,i,:,:] = X_test[:,i,:,:] - mean[i]
    newX_test[:,i,:,:] = newX_test[:,i,:,:] / std[i]
        
    
X_train = newX_train
X_test = newX_test

print "mean after normalization:", np.mean(X_train)
print "std after normalization:", np.std(X_train)

learningRate= 0.001               #-- Learning rate for the network
lr_weight_decay = np.array([10, 1, 0.1, 0.01])            #-- Learning weight decay. Reduce the learn rate by 0.95 after epoch

print "learning rate array size:", lr_weight_decay.size

class SGDLearningRateTracker(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations))
        print('\nLR: {:.6f}\n'.format(lr))

for i in xrange(lr_weight_decay.size):
    img_rows, img_cols = 32, 32       #-- input image dimensions
    num_classes = 10                  #-- Number of classes in CIFAR-10 dataset

    model = Sequential()                                                #-- Sequential container.
    model.add(Convolution2D(6, 5, 5,                                    #-- 6 outputs (6 filters), 5x5 convolution kernel
                        border_mode='valid',
                        input_shape=(3, img_rows, img_cols),dim_ordering="th"))       #-- 3 input depth (RGB)
    model.add(Activation('relu'))                                       #-- ReLU non-linearity 
    model.add(MaxPooling2D(pool_size=(2, 2),  dim_ordering="th"))                           #-- A max-pooling on 2x2 windows
    model.add(Convolution2D(16, 5, 5, dim_ordering="th"))                                  #-- 16 outputs (16 filters), 5x5 convolution kernel
    model.add(Activation('relu'))                                       #-- ReLU non-linearity
    model.add(MaxPooling2D(pool_size=(2, 2),  dim_ordering="th"))                           #-- A max-pooling on 2x2 windows

    model.add(Flatten())                                                #-- eshapes a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
    model.add(Dense(120))                                               #-- 120 outputs fully connected layer
    model.add(Activation('relu'))                                       #-- ReLU non-linearity 
    model.add(Dense(84))                                                #-- 84 outputs fully connected layer
    model.add(Activation('relu'))                                       #-- ReLU non-linearity 
    model.add(Dense(num_classes))                                       #-- 10 outputs fully connected layer (one for each class)
    model.add(Activation('softmax'))                                    #-- converts the output to a log-probability. Useful for classification problems
    print model.summary()

# specify training parameters
    batchSize = 50                    #-- Training Batch Size
    num_epochs = 10                   #-- Number of epochs for training   

    Y_train = np_utils.to_categorical(y_train, num_classes)
    Y_test = np_utils.to_categorical(y_test, num_classes)
    sgd = SGD(lr=learningRate, decay = lr_weight_decay[i])
    model.compile(loss='categorical_crossentropy',
                          optimizer='sgd',
                                        metrics=['accuracy'])

#    print(model.lr.get_value())
    #-- switch verbose=0 if you get error "I/O operation from closed file"
    history = model.fit(X_train, Y_train, batch_size=batchSize, nb_epoch=num_epochs,
                      verbose=1, shuffle=True, validation_data=(X_test, Y_test), callbacks=[SGDLearningRateTracker()])
#    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'], label='weight decay = ' + str(lr_weight_decay[i]))
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.hold(True)

plt.legend(loc='upper left')
plt.show()

