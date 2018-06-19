import argparse
parser = argparse.ArgumentParser(description='WGAN-GP')
parser.add_argument('--batch_size', type=int, default=32, required=False,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=1000, required=False,
                    help='epochs')
parser.add_argument('--preview_iteration', type=int, default=500, required=False,
                    help='preview_iteration')
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
from keras.datasets import mnist
from tools import *
from cvaegan import CVAEGAN
from skimage.io import imsave
from sklearn.utils import shuffle as skshuffle
from tqdm import tqdm

BS = args.batch_size
EPOCHS = args.epochs
w, h, c = 32, 32, 1
latent_dim = 100

(x_train, y_train), (___, __) = mnist.load_data()
x_train = np.squeeze(x_train.astype(np.float32)-127.5) / 127.5
x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant', constant_values=0)[...,np.newaxis]

y_train = keras.utils.to_categorical(y_train, 10)
N_CLASS = 10

if not os.path.exists('./preview'):
    os.makedirs('./preview')

trainer = CVAEGAN(input_shape=(h, w, c), num_attrs=N_CLASS, z_dims=latent_dim)  
generator = trainer.return_models()[1]

i_counter = 0
for epoch in range(EPOCHS):
    print("Epoch: %d / %d"%(epoch+1, EPOCHS))
    x_train, y_train = skshuffle(x_train, y_train)
    with tqdm(total=int(np.ceil(float(len(x_train)) / BS))) as t:
        for i in range(0, len(x_train), BS):
            r_bound = min(len(x_train), i+BS)
            l_bound = r_bound - BS
            image_batch = x_train[l_bound:r_bound]
            image_label = y_train[l_bound:r_bound]
            
            losses = trainer.train_on_batch((image_batch, image_label))
            
            if i_counter % args.preview_iteration == 0:
                generate_images_cvaegan(generator, './preview', h, w, c, latent_dim, 5, N_CLASS, i_counter)
                trainer.save_models('./weights', i_counter)
            i_counter += 1
            
            msg = 'g_loss: {:.2f}, d_loss: {:.2f}, c_loss: {:.2f}, e_loss: {:.2f}'.format(losses['g_loss'], losses['d_loss'], losses['c_loss'], losses['e_loss'])
            t.set_description(msg)
            t.update()
            
    trainer.save_models('./weights_epoch', epoch)
