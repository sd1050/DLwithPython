# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from keras.datasets import mnist


# %%
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# %%
print(str(train_images.shape) + '\n' + str(len(train_labels)) + '\n' + str(train_labels))


# %%
print(str(test_images.shape) + '\n' + str(len(test_labels)) + '\n' + str(test_labels))


# %%
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# network architecture, layers have data go in and come out in a more useful form extracting representations of the data fed into them


# %%
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# loss function - how the network can measure its performance on training data and adjust in the right direction
# optimizer - how the network will update itself based on data and loss function
# metrics - accuracy is fraction of images correctly classified


# %%
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32')/255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32')/255

# preprocessing data by reshaping it into what the network expects and scaling between [0 1] interval


# %%
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# %%
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# where we fit the model to its training data


# %%
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)


'''
digit = train_images[4]

import matplotlib.pyplot as plt 
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
'''

