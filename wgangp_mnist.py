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
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import keras
from keras import backend as K
K.set_session(session)
from keras.models import *
from tools import *
from models import , build_gan
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from skimage.io import imsave
from tqdm import tqdm

BS = args.batch_size
EPOCHS = args.epochs
w, h, c = 32, 32, 1
latent_dim = 100
D_ITER = 5
generator_model, discriminator_model, decoder, discriminator = build_gan(h=h, w=w, c=c, latent_dim=latent_dim, epsilon_std=args.std, batch_size=BS, dropout_rate=0.2)

(x_train, _), (___, __) = mnist.load_data()
x_train = np.squeeze(x_train.astype(np.float32)-127.5) / 127.5
x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant', constant_values=0)[...,np.newaxis]

if not os.path.exists('./preview'):
    os.makedirs('./preview')

d_counter = 0
i_counter = 0
for epoch in range(EPOCHS):
    print("Epoch: %d / %d"%(epoch+1, EPOCHS))
    np.random.shuffle(x_train)
    with tqdm(total=int(np.ceil(float(len(x_train)) / BS))) as t:
        for i in range(0, len(x_train), BS):
            r_bound = min(len(x_train), i+BS)
            l_bound = r_bound - BS
            image_batch = x_train[l_bound:r_bound]
            noise = np.random.normal(0, args.std, (BS, latent_dim)).astype(np.float32)
            msg = ''
            msg += 'DL: {:.2f}, '.format(np.mean(discriminator_model.train_on_batch([image_batch, noise], None)))
            d_counter += 1
            if d_counter==D_ITER:
                msg += 'GL: {:.2f}, '.format(np.mean(generator_model.train_on_batch(np.random.normal(0, args.std, (BS, latent_dim)).astype(np.float32), None)))
                d_counter = 0
            t.set_description(msg)
            t.update()
            if i_counter % 500 == 0:
                generate_images(decoder, './preview', h, w, c, latent_dim, args.std, 15, 15, i_counter, BS)
            i_counter += 1
    decoder.save('./decoder.h5')
    discriminator.save('./discriminator.h5')
