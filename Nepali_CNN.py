# -*- coding: utf-8 -*-
"""
Created on %(date)s
# =============================================================================
# 
# =============================================================================
@author: %(Drakael)s
"""

#print(__doc__)

from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from keras import models
from keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model

#fonction utile pour le tracing
def p(mess,obj):
    """Useful function for tracing"""
    if hasattr(obj,'shape'):
        print(mess,type(obj),obj.shape,"\n",obj)
    else:
        print(mess,type(obj),"\n",obj)

X = pd.read_csv('nepali_images.csv')
y = pd.read_csv('nepali_labels.csv')

X = np.array(X).reshape(len(X),len(X.columns))
y = np.array(y).reshape(len(y),len(y.columns))

n_row, n_col = 15, 20
n_components = n_row * n_col
image_shape = (36, 36)
# input image dimensions
img_rows, img_cols = image_shape

model_path = 'nepali.h5'


batch_size = 128
num_classes = len(np.unique(y))
epochs = 18

def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    
print(X.shape)
print(len(y))
print(y)   


   
#network = models.Sequential()
#network.add(layers.Dense(512, activation='relu', input_shape=(36 * 36,)))
#network.add(layers.Dense(len(np.unique(y)), activation='softmax'))
#network.compile(optimizer='rmsprop',
#                loss='categorical_crossentropy',
#                metrics=['accuracy'])




train_images, test_images, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=0)

train_images = train_images.reshape(len(train_images), 36 * 36)
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape(len(test_images), 36 * 36)
test_images = test_images.astype('float32') / 255
test_images_original = test_images
train_labels = to_categorical(train_labels)
test_labels_original = test_labels
test_labels = to_categorical(test_labels)

#plot_gallery("First centered number images", train_images[:n_components])

#network.fit(train_images, train_labels, epochs=5, batch_size=128)
#test_loss, test_acc = network.evaluate(test_images, test_labels)
#print('test_acc:', test_acc)

if K.image_data_format() == 'channels_first':
    train_images = train_images.reshape(train_images.shape[0], 1, img_rows, img_cols)
    test_images = test_images.reshape(test_images.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
    test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

print(model.summary())

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

print(model.summary())

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#model = load_model(model_path)

history = model.fit(train_images, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_images, test_labels))

def plot_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(loss) + 1)
    
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
      
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_history(history)

model.save(model_path)

predictions = model.predict(test_images, batch_size, verbose=0)
p('predictions',predictions)

p('test_labels_original',test_labels_original)
p('argmax predictions',np.argmax(predictions, axis=1))

#bad_predictions = test_images[predictions != test_labels_original]

#plot_gallery("First centered number images", bad_predictions[:n_components])

predicted_class = np.argmax(predictions, axis=1)
p('predicted_class',predicted_class)

mask = predicted_class!=np.squeeze(test_labels_original)
p('mask',mask)

p('test_images_original',test_images_original)
wrong_guesses_images = test_images_original[mask]

wrong_guesses_predictions = predictions[mask]

wrong_guesses_class = predicted_class[mask]

good_labels = test_labels_original[mask]

    
def plot_gallery_2(title, images, predicted_class , predictions, targets, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        idx_sort = np.argsort(predictions[i])[::-1]
        guess = predicted_class[i]
        first_guess = idx_sort[0]
        second_guess = idx_sort[1]
        third_guess = idx_sort[2]
        fourth_guess = idx_sort[3]
        true = targets[i]
        display = str(true)+'/'+str(first_guess)+'-'+str(second_guess)+'-'+str(third_guess)+'-'+str(fourth_guess)
        plt.title(display)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.15, 0.50)

plot_gallery_2("Wrong guesses", wrong_guesses_images, wrong_guesses_class, wrong_guesses_predictions, good_labels)





score = model.evaluate(test_images, test_labels, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
