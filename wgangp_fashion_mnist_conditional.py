import argparse
parser = argparse.ArgumentParser(description='Fashion Mnist Generation with GAN')
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
from models import wgangp_conditional
from keras.datasets import fashion_mnist as mnist
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from skimage.io import imsave
from sklearn.utils import shuffle as skshuffle
from tqdm import tqdm

BS = args.batch_size
EPOCHS = args.epochs
w, h, c = 32, 32, 1
latent_dim = 100
D_ITER = 5
generator_model, discriminator_model, classifier_model, generator, discriminator, classifier = wgangp_conditional(h=h, w=w, c=c, latent_dim=latent_dim, condition_dim=10, epsilon_std=args.std, dropout_rate=0.2)

(x_train, y_train), (___, __) = mnist.load_data()
x_train = np.squeeze(x_train.astype(np.float32)-127.5) / 127.5
x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant', constant_values=-1)[...,np.newaxis]
y_train = keras.utils.to_categorical(y_train, 10)

if not os.path.exists('./preview'):
    os.makedirs('./preview')

def make_some_noise():
    noise = np.random.normal(0, args.std, (BS, latent_dim)).astype(np.float32)
    condition = keras.utils.to_categorical(np.random.randint(10, size=(BS,)), 10)
    z = np.append(noise, condition, axis=-1)
    return z, condition

i_counter = 0
DL, CL, GL = 0, 0, 0
for epoch in range(EPOCHS):
    print("Epoch: %d / %d"%(epoch+1, EPOCHS))
    x_train, y_train = skshuffle(x_train, y_train)
    with tqdm(total=int(np.ceil(float(len(x_train)) / BS))) as t:
        for i in range(0, len(x_train), BS):
            r_bound = min(len(x_train), i+BS)
            l_bound = r_bound - BS
            image_batch = x_train[l_bound:r_bound]
            image_label = y_train[l_bound:r_bound]
            
            z, condition = make_some_noise()
            DL += np.mean(discriminator_model.train_on_batch([image_batch, z], None))
            CL += np.mean(classifier_model.train_on_batch(image_batch, image_label))
            
            if (i_counter+1) % D_ITER == 0:
                z, condition = make_some_noise()
                # CL += np.mean(classifier_model.train_on_batch(image_batch, image_label)) * D_ITER
                GL += np.mean(generator_model.train_on_batch(z, condition)) * D_ITER
                
            if i_counter % 500 == 0:
                generate_images_cgan(generator, './preview', h, w, c, latent_dim, args.std, 5, 10, i_counter+1)
            i_counter += 1
            
            msg = 'DL: {:.2f}, CL: {:.2f}, GL: {:.2f}'.format(DL/i_counter, CL/i_counter, GL/i_counter) # running mean
            t.set_description(msg)
            t.update()
            
    generator.save('./generator.h5')
    discriminator.save('./discriminator.h5')
    classifier.save('./classifier.h5')
