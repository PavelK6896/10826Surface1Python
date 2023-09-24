import io

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageEnhance

print(tf.__version__)
# https://github.com/keras-team/keras-core/issues/855
model = tf.keras.saving.load_model(filepath='model/mnist-1-1695561596')

fill_color = (252, 252, 252)


def test():
    img_path = 'new/90b6f012-48e8-4ed2-ab1f-cb0be0ea2dca.png'

    image = Image.open(img_path)
    im = image.convert("RGBA")
    if im.mode in ('RGBA', 'LA'):
        background = Image.new(im.mode[:-1], im.size, fill_color)
        background.paste(im, im.split()[-1])
        im = background

    im.save('rr1.png')
    rr = im.convert("L")
    rr = ImageOps.invert(rr)
    rr.save('rr2.png')
    rr = rr.resize((28, 28))
    rr = ImageEnhance.Contrast(rr).enhance(2)
    rr.save('rr3.png')

    img = np.array(rr)
    image = img.reshape(1, 28, 28, 1)
    print(image.shape)

    model = tf.keras.saving.load_model('model/mnist-1.keras')
    predict1 = model.predict(image)
    print(predict1)
    r = predict1[0].argmax(axis=0)
    print(r)


def adapt(image):
    image = Image.open(io.BytesIO(image))
    im = image.convert("RGBA")
    if im.mode in ('RGBA', 'LA'):
        background = Image.new(im.mode[:-1], im.size, fill_color)
        background.paste(im, im.split()[-1])
        im = background

    rr = im.convert("L")
    rr = ImageOps.invert(rr)
    rr = rr.resize((28, 28))
    rr = ImageEnhance.Contrast(rr).enhance(2)
    return rr


def predict(data):
    img = np.array(data)
    image = img.reshape(1, 28, 28, 1)
    predict1 = model.predict(image)
    r = predict1[0].argmax(axis=0)
    return r
