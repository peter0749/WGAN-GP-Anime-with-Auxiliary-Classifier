import argparse
parser = argparse.ArgumentParser(description='Music Generation with VAE')
parser.add_argument('--batch_size', type=int, default=32, required=False,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=120, required=False,
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
from vae_model import build_residual_vae, build_vae_gan
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from skimage.io import imsave
from tqdm import tqdm

BS = args.batch_size
EPOCHS = args.epochs
w, h, c = 96, 96, 3
latent_dim = 100
D_ITER = 5
generator_model, discriminator_model, decoder, discriminator = build_vae_gan(h=h, w=w, c=c, latent_dim=latent_dim, epsilon_std=args.std, batch_size=BS, dropout_rate=0.2, use_vae=False)

train_generator = data_generator('./anime-faces', height=h, width=w, channel=3, batch_size=BS, shuffle=True, normalize=False)
seq = get_imgaug()

if not os.path.exists('./preview'):
    os.makedirs('./preview')

d_counter = 0
for epoch in range(EPOCHS):
    print("Epoch: %d / %d"%(epoch+1, EPOCHS))
    train_generator.random_shuffle()
    with tqdm(total=len(train_generator)) as t:
        for i in range(len(train_generator)):
            image_batch = train_generator.__getitem__(i)
            image_batch = seq.augment_images(image_batch)
            image_batch = (image_batch.astype(np.float32) - 127.5) / 127.5
            noise = np.random.normal(0, args.std, (BS, latent_dim)).astype(np.float32)
            msg = ''
            msg += 'DL: {:.2f}, '.format(np.mean(discriminator_model.train_on_batch([image_batch, noise], None)))
            d_counter += 1
            if d_counter==D_ITER:
                msg += 'GL: {:.2f}, '.format(np.mean(generator_model.train_on_batch(np.random.normal(0, args.std, (BS, latent_dim)).astype(np.float32), None)))
                d_counter = 0
            t.set_description(msg)
            t.update()
    generate_images(decoder, './preview', h, w, c, latent_dim, args.std, 15, 15, epoch, BS)
    decoder.save('./decoder.h5')
    discriminator.save('./discriminator.h5')
