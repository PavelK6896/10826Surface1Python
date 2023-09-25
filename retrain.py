import os
import random
import time

import matplotlib.pyplot as plt
import numpy
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils

print(tf.__version__)

training = '..\\mnist_png\\training'
testing = '..\\mnist_png\\testing'
new = '.\\new'
### new

x_new = []
y_new = []
id1 = 0
images = sorted(os.listdir(new))
for num, img in enumerate(images):
    print(num, img)
    f = new + '\\' + str(num)
    for num2, num2 in enumerate(os.listdir(f)):
        x_new.append(numpy.array(Image.open(f + '\\' + num2)))
        y_new.append(num)
        id1 = id1 + 1
        # if(id1 > 10):
        #     break
print(id1)

x_new = numpy.array(x_new)
y_new = numpy.array(y_new)

print(x_new.shape)
print(y_new.shape)

label_indexes = np.where(y_new == 3)[0]
label_indexes2 = np.where(y_new == 2)[0]
label_indexes6 = np.where(y_new == 6)[0]
print(label_indexes)
print(label_indexes.size)
print(label_indexes2.size)
print(label_indexes6.size)

x_new = x_new.reshape(x_new.shape[0], x_new.shape[1], x_new.shape[2], 1)
y_new = utils.to_categorical(y_new, 10)

fig, axs = plt.subplots(1, 10, figsize=(25, 3))
for i in range(10):
    index = random.randint(0, id1 - 1)
    img = x_new[index]
    n = y_new[index]
    print(img.shape)
    print(n.shape)
    axs[i].imshow(Image.fromarray(img.reshape(img.shape[0], img.shape[1])), cmap='gray')
    axs[i].text(15, 35, str(n.argmax(axis=0)), bbox=dict(fill=False, edgecolor='red', linewidth=2))

plt.show()

print(x_new.shape)
print(y_new.shape)

### training

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

### test
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

### train_test_split

print(x_new.shape)
print(x_test.shape)

x_new2 = np.concatenate([x_new, x_test])
print(x_new2.shape)
x_new3 = np.concatenate([x_new2, x_train])
print(x_new3.shape)

y_new2 = np.concatenate([y_new, y_test])
y_new3 = np.concatenate([y_new2, y_train])

print(x_new3.shape)
print(y_new3.shape)

print(x_new3.shape[0])
print(y_new3.shape[0])

X_train, X_test, Y_train, Y_test = train_test_split(x_new3, y_new3, stratify=y_new3, test_size=0.1, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(X_train.shape[0])
print(X_test.shape[0])

xs = X_train.shape[0]

fig, axs = plt.subplots(1, 20, figsize=(25, 3))
for i in range(20):
    index = random.randint(0, xs - 1)
    img = X_train[index]
    n = Y_train[index]
    print(img.shape)
    print(n.shape)
    axs[i].imshow(Image.fromarray(img.reshape(img.shape[0], img.shape[1])), cmap='gray')
    axs[i].text(15, 35, str(n.argmax(axis=0)), bbox=dict(fill=False, edgecolor='red', linewidth=2))

plt.show()

m = 'model/mnist-1-1695634354'
print("load model" + m)
model = tf.keras.models.load_model(m)

model_Checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='model/retrain/mnist-1-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}', monitor='val_loss',
    verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

history = model.fit(X_train, Y_train, batch_size=128, epochs=25, validation_data=(X_test, Y_test), verbose=1,
                    callbacks=[model_Checkpoint])

print(history.history['accuracy'])
print(history.history['val_accuracy'])

plt.plot(history.history['accuracy'], label='correct answers training')
plt.plot(history.history['val_accuracy'], label='correct answers testing')
plt.xlabel('epoch')
plt.ylabel('correct answers')
plt.legend()
plt.show()

ts = int(time.time())
file_path = f"model/mnist-1-{ts}"
tf.keras.saving.save_model(model=model, filepath=file_path, save_format="tf")
# https://github.com/keras-team/keras-core/issues/855
