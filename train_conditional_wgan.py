import argparse
parser = argparse.ArgumentParser(description='WGAN-GP')
parser.add_argument('--dataset', type=str, required=True,
                    help='path to dataset')
parser.add_argument('--load_weights', action='store_true', default=False,
                    help='continue training')
parser.add_argument('--width', type=int, default=96, required=False,
                    help='width')
parser.add_argument('--height', type=int, default=96, required=False,
                    help='height')
parser.add_argument('--channels', type=int, default=3, required=False,
                    help='channels')
parser.add_argument('--z_dim', type=int, default=100, required=False,
                    help='latent dimension')
parser.add_argument('--batch_size', type=int, default=32, required=False,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=1000, required=False,
                    help='epochs')
parser.add_argument('--preview_iteration', type=int, default=500, required=False,
                    help='preview_iteration')
parser.add_argument('--std', type=float, default=1.0, required=False,
                    help='sampling std')
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
from models import wgangp_conditional
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from skimage.io import imsave
from tqdm import tqdm

use_data_augmentation = not args.no_augmentation

BS = args.batch_size
EPOCHS = args.epochs
w, h, c = args.width, args.height, args.channels
latent_dim = args.z_dim
D_ITER = 5

train_generator = data_generator(args.dataset, height=h, width=w, channel=c, batch_size=BS, shuffle=True, normalize=not use_data_augmentation, save_tags=True)
N_CLASS = len(train_generator.tags)
print('This dataset has %d unique tags'%N_CLASS)
generator_model, discriminator_model, classifier_model, generator, discriminator, classifier = wgangp_conditional(h=h, w=w, c=c, latent_dim=latent_dim, condition_dim=N_CLASS , epsilon_std=args.std, batch_size=BS, dropout_rate=0.2)

seq = get_imgaug()

if args.load_weights:
    generator.load_weights('./generator.h5')
    discriminator.load_weights('./discriminator.h5')
    classifier.load_weights('./classifier.h5')

if not os.path.exists('./preview'):
    os.makedirs('./preview')
    
def make_some_noise():
    noise = np.random.normal(0, args.std, (BS, latent_dim)).astype(np.float32)
    condition = keras.utils.to_categorical(np.random.randint(N_CLASS, size=(BS,)), N_CLASS)
    z = np.append(noise, condition, axis=-1)
    return z, condition

i_counter = 0
DL, CL, GL = 0, 0, 0
for epoch in range(EPOCHS):
    print("Epoch: %d / %d"%(epoch+1, EPOCHS))
    train_generator.random_shuffle()
    with tqdm(total=len(train_generator)) as t:
        for i in range(len(train_generator)):
            image_batch, image_label = train_generator.__getitem__(i)
            if use_data_augmentation:
                image_batch = seq.augment_images(image_batch)
                image_batch = (image_batch.astype(np.float32) - 127.5) / 127.5
            
            z, condition = make_some_noise()
            DL += np.mean(discriminator_model.train_on_batch([image_batch, z], None))
            CL += np.mean(classifier_model.train_on_batch(image_batch, image_label))
            
            if (i_counter+1) % D_ITER == 0:
                z, condition = make_some_noise()
                # CL += np.mean(classifier_model.train_on_batch(image_batch, image_label)) * D_ITER
                GL += np.mean(generator_model.train_on_batch(z, condition)) * D_ITER
            
            if i_counter % args.preview_iteration == 0:
                generate_images_cgan(generator, './preview', h, w, c, latent_dim, args.std, 5, N_CLASS, i_counter+1)
            i_counter += 1
            
            msg = 'DL: {:.2f}, CL: {:.2f}, GL: {:.2f}'.format(DL/i_counter, CL/i_counter, GL/i_counter) # running mean
            t.set_description(msg)
            t.update()
            
    generator.save('./generator.h5')
    discriminator.save('./discriminator.h5')
    classifier.save('./classifier.h5')
