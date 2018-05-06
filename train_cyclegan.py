import argparse
parser = argparse.ArgumentParser(description='WGAN-GP')
parser.add_argument('--dataset_A', type=str, required=True,
                    help='path to dataset A')
parser.add_argument('--dataset_B', type=str, required=True,
                    help='path to dataset B')
parser.add_argument('--width', type=int, default=96, required=False,
                    help='width')
parser.add_argument('--height', type=int, default=96, required=False,
                    help='height')
parser.add_argument('--channels_A', type=int, default=3, required=False,
                    help='channels of A')
parser.add_argument('--channels_B', type=int, default=3, required=False,
                    help='channels of B')
parser.add_argument('--batch_size', type=int, default=8, required=False,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=1000, required=False,
                    help='epochs')
parser.add_argument('--no_augmentation', action='store_true', default=False,
                    help='')
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
from vae_model import build_cyclewgan
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from skimage.io import imsave
from tqdm import tqdm

use_data_augmentation = not args.no_augmentation

BS = args.batch_size
EPOCHS = args.epochs
w, h, c_A, c_B = args.width, args.height, args.channels_A, args.channels_B
D_ITER = 5
generator_model, discriminator_A_model, discriminator_B_model, generator_A, generator_B, discriminator_A, discriminator_B = build_cyclewgan(h=h, w=w, c_A=c_A, c_B=c_B, batch_size=BS, dropout_rate=0.2)

train_generator_A = data_generator(args.dataset_A, height=h, width=w, channel=c_A, batch_size=BS, shuffle=True, normalize=not use_data_augmentation)
train_generator_B = data_generator(args.dataset_B, height=h, width=w, channel=c_B, batch_size=BS, shuffle=True, normalize=not use_data_augmentation)
seq = get_imgaug()

if not os.path.exists('./preview'):
    os.makedirs('./preview')

d_counter = 0
for epoch in range(EPOCHS):
    print("Epoch: %d / %d"%(epoch+1, EPOCHS))
    train_generator_A.random_shuffle()
    train_generator_B.random_shuffle()
    min_len = min(len(train_generator_A), len(train_generator_B))
    with tqdm(total=min_len) as t:
        for i in range(min_len):
            image_batch_A, _ = train_generator_A.__getitem__(i)
            image_batch_B, _ = train_generator_B.__getitem__(i)
            if use_data_augmentation:
                image_batch_A = seq.augment_images(image_batch_A)
                image_batch_B = seq.augment_images(image_batch_B)
                image_batch_A = (image_batch_A.astype(np.float32) - 127.5) / 127.5
                image_batch_B = (image_batch_B.astype(np.float32) - 127.5) / 127.5
            msg = ''
            msg += 'DL_A: {:.2f}, '.format(np.mean(discriminator_A_model.train_on_batch([image_batch_A, image_batch_B], None)))
            msg += 'DL_B: {:.2f}, '.format(np.mean(discriminator_B_model.train_on_batch([image_batch_B, image_batch_A], None)))
            d_counter += 1
            if d_counter==D_ITER:
                msg += 'GL: {:.2f}, '.format(np.mean(generator_model.train_on_batch([image_batch_A, image_batch_B], None))) # train A->B, B->A
                d_counter = 0
            t.set_description(msg)
            t.update()
    generate_images_cyclegan(generator_A, generator_B, train_generator_A.__getitem__(0)[0][0], train_generator_B.__getitem__(0)[0][0], './preview', h, w, c_A, c_B, epoch)
    generator_A.save('./generator_A.h5')
    generator_B.save('./generator_B.h5')
    discriminator_A.save('./discriminator_A.h5')
    discriminator_B.save('./discriminator_B.h5')