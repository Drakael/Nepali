# -*- coding: utf-8 -*-
"""
Created on %(date)s
# =============================================================================
# 
# =============================================================================
@author: %(Drakael)s
"""

#print(__doc__)

#from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
#from keras import models
#from keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from keras.models import Model

from cntk.device import try_set_default_device, gpu, get_gpu_properties
try_set_default_device(gpu(0), acquire_device_lock=True)

print(get_gpu_properties(gpu(0)))


#little usefull function for tracing
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

print(X.shape)
print(len(y))
print(y)  

# input image dimensions
image_shape = (36, 36)
img_rows, img_cols = image_shape

model_path = 'nepali.h5'
temp_path = 'temp.h5'

batch_size = 32
num_classes = len(np.unique(y))
epochs = 16

def plot_gallery(title, images, image_shape):
    p('images',images)
    n_col = int(np.ceil(np.sqrt(images.shape[0])))
    n_row = int(np.ceil(np.sqrt(images.shape[0])))
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images[:(n_col*n_row)]):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


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
   
train_images, test_images, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=0)

train_images = train_images.reshape(len(train_images), 36 * 36)
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape(len(test_images), 36 * 36)
test_images = test_images.astype('float32') / 255
test_images_original = test_images
train_labels = to_categorical(train_labels)
test_labels_original = test_labels
test_labels = to_categorical(test_labels)

if K.image_data_format() == 'channels_first':
    train_images = train_images.reshape(train_images.shape[0], 1, img_rows, img_cols)
    test_images = test_images.reshape(test_images.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
    test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


#model = Sequential()
#model.add(Conv2D(64, kernel_size=(7, 7),
#                 activation='relu',
#                 padding='same',
#                 input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Conv2D(64, (9, 9), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(num_classes, activation='softmax'))

#model = load_model(model_path)
model = load_model('nepali_64x7x7sP_64x9x9sP_64x5x5sP_512D_967.h5')

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


print(model.summary())

#history = model.fit(train_images, train_labels,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          validation_data=(test_images, test_labels))
#
#
#plot_history(history)
#
#model.save(model_path)
#model.save(temp_path)

predictions = model.predict(test_images, batch_size, verbose=0)
p('predictions',predictions)

p('test_labels_original',test_labels_original)
p('argmax predictions',np.argmax(predictions, axis=1))


predicted_class = np.argmax(predictions, axis=1)
p('predicted_class',predicted_class)

mask = predicted_class!=np.squeeze(test_labels_original)
p('mask',mask)

p('test_images_original',test_images_original)
wrong_guesses_images = test_images_original[mask]

wrong_guesses_predictions = predictions[mask]

wrong_guesses_class = predicted_class[mask]

good_labels = test_labels_original[mask]

    
def plot_gallery_2(title, images, image_shape, predicted_class=None , predictions=None, targets=None, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images[:(n_col*n_row)]):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        idx_sort = np.argsort(predictions[i])[::-1]
        if predicted_class is not None and predictions is not None and targets is not None:
            #guess = predicted_class[i]
#            first_guess = idx_sort[0]
#            second_guess = idx_sort[1]
#            third_guess = idx_sort[2]
#            fourth_guess = idx_sort[3]
            true = targets[i]
            guess_rank = idx_sort.tolist().index(true)
            #display = str(true)+'/'+str(first_guess)+'-'+str(second_guess)+'-'+str(third_guess)+'-'+str(fourth_guess)
            display = str(true)+' / '+str(guess_rank)
            plt.title(display)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.25, 0.50)

plot_gallery_2("Wrong guesses", wrong_guesses_images, image_shape, wrong_guesses_class, wrong_guesses_predictions, good_labels)


score = model.evaluate(test_images, test_labels, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[1].output)
intermediate_output = intermediate_layer_model.predict(test_images)

weights = np.array(model.get_weights())
for w in weights:
    print(w.shape)
    
cur_weights = weights[0]
weight_map = np.moveaxis(cur_weights, [0, 1, 2, 3], [2, 3, 1, 0]).reshape(-1, cur_weights.shape[0], cur_weights.shape[1])
plot_gallery("conv1 weights", weight_map, (cur_weights.shape[0], cur_weights.shape[1]))

cur_weights = weights[2]
weight_map = np.moveaxis(cur_weights, [0, 1, 2, 3], [2, 3, 1, 0]).reshape(-1, cur_weights.shape[0], cur_weights.shape[1])
plot_gallery("conv2 weights", weight_map, (cur_weights.shape[0], cur_weights.shape[1]))

cur_weights = weights[4]
weight_map = np.moveaxis(cur_weights, [0, 1, 2, 3], [2, 3, 1, 0]).reshape(-1, cur_weights.shape[0], cur_weights.shape[1])
plot_gallery("conv3 weights", weight_map, (cur_weights.shape[0], cur_weights.shape[1]))



def plot_image(image, image_shape, n_col, n_row, index):
    if index <= n_col * n_row:
        plt.subplot(n_row, n_col, index)
        vmax = max(image.max(), -image.min())
        plt.imshow(image.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())

def plot_layers_weights(title, model):
    weights = np.array(model.get_weights())
    weight_map = []
    weight_shape = []
    bias = []
    
    for i, w in enumerate(weights):
        print(w.shape)
        if(len(w.shape)==4):
            weight_shape.append(w.shape)
            movaxis = np.moveaxis(w, [0, 1, 2, 3], [3, 2, 0, 1])
            add_weights = movaxis.reshape(-1,movaxis.shape[2],movaxis.shape[3])
            weight_map.append(add_weights)
        elif(len(w.shape)==1):
            add_bias = w 
            bias.append(add_bias)  
    
    if(len(weight_map)>1):
        nb_kernel_layer_1 = weight_shape[1][3]
        n_col = nb_kernel_layer_1 + 2
        n_row = 0.5 * n_col
        print('n_col, n_row = ',n_col,', ',n_row)
    else:
        nb_kernel_layer_0 = weight_shape[0][3]
        n_col = nb_kernel_layer_0
        n_row = 0.5 * n_col
        print('n_col, n_row = ',n_col,', ',n_row)
        
            
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    grid_size = n_col * n_row
    cnt = 0
    p('grid_size',grid_size)
    for i, layer_0 in enumerate(weight_map[0]):
        if(len(weight_map)>1):
            plot_image(layer_0, layer_0.shape, n_col, n_row, (i*n_col) + 1)
            cnt+=2
            for j, layer_1 in enumerate(weight_map[1][ i *nb_kernel_layer_1 : ((i+1) * nb_kernel_layer_1) ]):
                plot_image(layer_1, layer_1.shape, n_col, n_row, (i*n_col) + 3 + j)
                cnt+=1
        else:
            plot_image(layer_0, layer_0.shape, n_col, n_row, i + 1)
            cnt+=1
    
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.2, 0.1)
    
plot_layers_weights('Features weights', model)