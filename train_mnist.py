import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import *
from tools import *
from vae_model import build_inception_residual_vae
from keras.datasets import mnist
from keras.callbacks import Callback
from skimage.io import imsave

w, h, c = 32, 32, 1
BS = 20
EPOCHS = 10

vae, encoder, decoder = build_inception_residual_vae(h=h, w=w, c=c, latent_dim=2, epsilon_std=1., dropout_rate=0.2)

(x_train, _), (x_test, __) = mnist.load_data()
x_train = (np.pad(x_train, [(0,0), (2,2), (2,2)], 'constant')[...,np.newaxis] - 127.5) / 127.5
x_test = (np.pad(x_test, [(0,0), (2,2), (2,2)], 'constant')[...,np.newaxis] - 127.5) / 127.5

if not os.path.exists('./mnist_preview'):
    os.makedirs('./mnist_preview')

vae.fit(x_train, None, validation_data=(x_test, None), batch_size=BS, epochs=EPOCHS, shuffle=True, callbacks=[Preview(decoder, './mnist_preview', h, w, std=1.)])
decoder.save('./mnist_decoder.h5')
encoder.save('./mnist_encoder.h5')