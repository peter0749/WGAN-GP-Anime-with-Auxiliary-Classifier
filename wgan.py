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
from vae_model import build_inception_residual_vae, build_vae_gan
from keras.datasets import mnist
from keras.callbacks import Callback
from skimage.io import imsave
from tqdm import tqdm

BS = args.batch_size
EPOCHS = args.epochs
w, h, c = 96, 96, 3
generator_model, discriminator_model, decoder, discriminator = build_vae_gan(h=h, w=w, c=c, latent_dim=2, epsilon_std=args.std, batch_size=BS, dropout_rate=0.2, use_vae=False)

train_generator = data_generator('./anime-faces', height=h, width=w, channel=c, batch_size=BS, shuffle=True)

if not os.path.exists('./preview'):
    os.makedirs('./preview')

for epoch in range(EPOCHS):
    print("Epoch: %d / %d"%(epoch+1, EPOCHS))
    with tqdm(total=len(train_generator)) as t:
        for i in range(len(train_generator)):
            image_batch, _ = train_generator.__getitem__(i)
            noise = np.random.normal(0, args.std, (BS, 2)).astype(np.float32)
            t.write('DL: {:.2f}, '.format(np.mean(discriminator_model.train_on_batch([image_batch, noise], None))))
            t.write('GL: {:.2f}, '.format(np.mean(generator_model.train_on_batch(np.random.normal(0, args.std, (BS, 2)).astype(np.float32), None))))
            t.update()
    train_generator.random_shuffle()
    generate_images(decoder, './preview', h, w, c, 1.0, 15, 15, epoch, BS)
    decoder.save('./decoder.h5')
    discriminator.save('./discriminator.h5')
