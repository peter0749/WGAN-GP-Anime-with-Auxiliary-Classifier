import argparse
parser = argparse.ArgumentParser(description='Music Generation with VAE')
parser.add_argument('--batch_size', type=int, default=16, required=False,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=50, required=False,
                    help='epochs')
parser.add_argument('--std', type=float, default=1.0, required=False,
                    help='sampling std')
args = parser.parse_args()

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import *
from tools import *
from vae_model import build_inception_residual_vae
from keras.datasets import mnist
from keras.callbacks import Callback
from skimage.io import imsave

w, h, c = 96, 96, 3
BS = args.batch_size
EPOCHS = args.epochs
vae, encoder, decoder = build_inception_residual_vae(h=h, w=w, c=c, latent_dim=2, epsilon_std=args.std, dropout_rate=0.2)

# train_generator = data_generator('./anime-faces', height=h, width=w, batch_size=BS, shuffle=True)
train_generator = data_generator('./f', height=h, width=w, batch_size=BS, shuffle=True)

if not os.path.exists('./preview'):
    os.makedirs('./preview')

vae.fit_generator(train_generator, epochs=EPOCHS, shuffle=True, workers=3, callbacks=[Preview(decoder, './preview', h, w, std=args.std)])
decoder.save('./decoder.h5')
encoder.save('./encoder.h5')
