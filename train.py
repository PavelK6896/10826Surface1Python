import numpy as np
import tensorflow as tf
import keras
import os
from PIL import Image
import numpy
from tensorflow.keras import utils

training = '..\\mnist_png\\training'
testing = '..\\mnist_png\\testing'

x_train = []
y_train = []
id1 = 0
images = sorted(os.listdir(training))
for num, img in enumerate(images):
    print(num, img)
    f = training + '\\' + str(num)
    for num2, num2 in enumerate(os.listdir(f)):
        x_train.append(numpy.array(Image.open(f + '\\' + num2)))
        y_train.append(num)
        id1 = id1 + 1
        # if(id1 > 10):
        #     break
print(id1)

x_train = numpy.array(x_train)
y_train = numpy.array(y_train)

print(x_train.shape)
print(y_train.shape)

label_indexes = np.where(y_train == 3)[0]
label_indexes2 = np.where(y_train == 2)[0]
label_indexes6 = np.where(y_train == 6)[0]
print(label_indexes)
print(label_indexes.size)
print(label_indexes2.size)
print(label_indexes6.size)

x_test = []
y_test = []
id1 = 0
images = sorted(os.listdir(testing))
for num, img in enumerate(images):
    print(num, img)
    f = testing + '\\' + str(num)
    for num2, num2 in enumerate(os.listdir(f)):
        x_test.append(numpy.array(Image.open(f + '\\' + num2)))
        y_test.append(num)
        id1 = id1 + 1
        # if(id1 > 10):
        #     break
print(id1)

x_test = numpy.array(x_test)
y_test = numpy.array(y_test)

print(x_test.shape)
print(y_test.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

y_train = utils.to_categorical(y_train, 10)

label_indexes = np.where(y_train == 3)[0]
label_indexes2 = np.where(y_train == 2)[0]
label_indexes6 = np.where(y_train == 6)[0]
print(label_indexes)
print(label_indexes.size)
print(label_indexes2.size)
print(label_indexes6.size)

y_test = utils.to_categorical(y_test, 10)

print('x_train:', x_train.shape)
print('x_test:', x_test.shape)
print()
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)

import matplotlib.pyplot as plt
import random
from PIL import Image

fig, axs = plt.subplots(1, 10, figsize=(25, 3))
for i in range(10):
    index = random.randint(0, 60_000)
    img = x_train[index]
    n = y_train[index]
    print(img.shape)
    print(n.shape)
    axs[i].imshow(Image.fromarray(img.reshape(img.shape[0], img.shape[1])), cmap='gray')
    axs[i].text(15, 35, str(n.argmax(axis=0)), bbox=dict(fill=False, edgecolor='red', linewidth=2))

plt.show()

model = keras.models.load_model('model/mnist-1.keras')
history = model.fit(x_train, y_train, batch_size=128, epochs=2, validation_data=(x_test, y_test), verbose=1)

print(history.history['accuracy'])
print(history.history['val_accuracy'])

model.save('model/mnist-2.keras')
