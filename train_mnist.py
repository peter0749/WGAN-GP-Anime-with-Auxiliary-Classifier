import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import keras
from keras import backend as K
K.set_session(session)
from keras.models import *
from keras.preprocessing.image import ImageDataGenerator
from tools import *
from vae_model import build_inception_residual_vae
from keras.datasets import mnist
from keras.callbacks import Callback
from skimage.io import imsave

w, h, c = 32, 32, 1
BS = 60
EPOCHS = 10

vae, encoder, decoder = build_inception_residual_vae(h=h, w=w, c=c, latent_dim=2, epsilon_std=1., dropout_rate=0.2)

(x_train, _), (x_test, __) = mnist.load_data()
train_generator = mnist_generator(x_train, w, h, BS)
valid_generator = mnist_generator(x_test,  w, h, BS)

if not os.path.exists('./mnist_preview'):
    os.makedirs('./mnist_preview')

vae.fit_generator(train_generator, validation_data=valid_generator, epochs=EPOCHS, shuffle=True, callbacks=[Preview(decoder, './mnist_preview', h, w, std=1., batch_size=BS)], workers=3)
decoder.save('./mnist_decoder.h5')
encoder.save('./mnist_encoder.h5')