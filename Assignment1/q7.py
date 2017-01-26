import numpy as np
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import theano
import keras
import scipy
from math import *
from keras.datasets import cifar10
from keras.models  import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
#%matplotlib inline

# load cifar dataset
cifar10 = np.load('../../../data/lab2/lab2_data/cifar10_data.npz')
X_train = cifar10['X_train']
y_train = cifar10['y_train']
X_test = cifar10['X_test']
y_test = cifar10['y_test']
num_classes = 10  


# adds salt and pepper noise to data
def noisy(image):
    row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.004
    out = image
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coordsy = [np.random.randint(0, image.shape[0] - 1, int(num_salt))]
    coordsx = [np.random.randint(0, image.shape[1] - 1, int(num_salt))]        
    out[coordsx, coordsy, :] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coordsy = [np.random.randint(0, image.shape[0] - 1, int(num_pepper))]
    coordsx = [np.random.randint(0, image.shape[1] - 1, int(num_pepper))]        
    out[coordsx, coordsy, :] = 0
    return out

# def rotate_image(image):


def create_noisy_data(X_train, y_train):
    no_noisy_samples = 10000   
    X_train_noisy = np.zeros((X_train.shape[0] + 2 * no_noisy_samples, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    y_train_noisy = np.zeros((y_train.shape[0] + 2 * no_noisy_samples, 1), dtype=np.int)
    noisy_coords = np.random.randint(0, X_train.shape[0] - 1, no_noisy_samples)
    rotated_coords = np.random.randint(0, X_train.shape[0] - 1, no_noisy_samples)
    for i in xrange(no_noisy_samples):
        plot_image1 = X_train[noisy_coords[i],:,:,:].transpose(1,2,0) 
        plot_image2 = X_train[rotated_coords[i],:,:,:].transpose(1,2,0)
        noisy_image = noisy(plot_image1)
        rotated_image = scipy.ndimage.interpolation.rotate(plot_image2, 90)
        X_train_noisy[2 * i,:,:,:] = noisy_image.transpose(2,0,1)
        X_train_noisy[2 * i + 1, :,:,:] = rotated_image.transpose(2,0,1)
        y_train_noisy[2 * i] = y_train[noisy_coords[i]]
        y_train_noisy[2 * i + 1] = y_train[rotated_coords[i]]

    X_train_noisy[2 * no_noisy_samples:,:,:,:] = X_train
    y_train_noisy[2 * no_noisy_samples:, :] = y_train
    permut = np.random.permutation(X_train_noisy.shape[0])
    X_train_noisy = X_train_noisy[permut,:,:,:]
    y_train_noisy = y_train_noisy[permut,:]
    cifar10_data_noisy = TemporaryFile()
    np.savez('../../../data/lab2/lab2_data/cifar10_data_noisy', X_train_noisy, y_train_noisy)

create_noisy_data(X_train, y_train)
cifar10 = np.load('../../../data/lab2/lab2_data/cifar10_data_noisy.npz')
X_train = cifar10['arr_0']
y_train = cifar10['arr_1']


print "Training data:"
print "Number of examples: ", X_train.shape[0]
print "Number of channels:",X_train.shape[1] 
print "Image size:", X_train.shape[2], X_train.shape[3]
print
print "Test data:"
print "Number of examples:", X_test.shape[0]
print "Number of channels:", X_test.shape[1]
print "Image size:",X_test.shape[2], X_test.shape[3] 


# # normalise data
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

batchSize = 50                    #-- Training Batch Size                #-- Number of classes in CIFAR-10 dataset
num_epochs = 10                   #-- Number of epochs for training   
learningRate= 0.001               #-- Learning rate for the network
lr_weight_decay = 0.95            #-- Learning weight decay. Reduce the learn rate by 0.95 after epoch


img_rows, img_cols = 32, 32       #-- input image dimensions

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

model = Sequential()                                                #-- Sequential container.

model.add(Convolution2D(6, 5, 5,                                    #-- 6 outputs (6 filters), 5x5 convolution kernel
                        border_mode='valid',
                        input_shape=(3, img_rows, img_cols)))       #-- 3 input depth (RGB)
model.add(Activation('relu'))                                       #-- ReLU non-linearity 
model.add(MaxPooling2D(pool_size=(2, 2)))                           #-- A max-pooling on 2x2 windows
model.add(Convolution2D(16, 5, 5))                                  #-- 16 outputs (16 filters), 5x5 convolution kernel
model.add(Activation('relu'))                                       #-- ReLU non-linearity
model.add(MaxPooling2D(pool_size=(2, 2)))                           #-- A max-pooling on 2x2 windows

model.add(Flatten())                                                #-- eshapes a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
model.add(Dense(120))                                               #-- 120 outputs fully connected layer
model.add(Activation('relu'))                                       #-- ReLU non-linearity 
model.add(Dense(84))                                                #-- 84 outputs fully connected layer
model.add(Activation('relu'))                                       #-- ReLU non-linearity 
model.add(Dense(num_classes))                                       #-- 10 outputs fully connected layer (one for each class)
model.add(Activation('softmax'))                                    #-- converts the output to a log-probability. Useful for classification problems

print model.summary()

sgd = SGD(lr=learningRate, decay = lr_weight_decay)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

#-- switch verbose=0 if you get error "I/O operation from closed file"
history = model.fit(X_train, Y_train, batch_size=batchSize, nb_epoch=num_epochs,
          verbose=1, shuffle=True, validation_data=(X_test, Y_test))

#-- summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#-- summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#-- test the network
score = model.evaluate(X_test, Y_test, verbose=0)

print 'Test score:', score[0] 
print 'Test accuracy:', score[1]




