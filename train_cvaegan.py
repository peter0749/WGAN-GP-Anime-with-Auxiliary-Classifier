import argparse
parser = argparse.ArgumentParser(description='WGAN-GP')
parser.add_argument('--dataset', type=str, required=True,
                    help='path to dataset')
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
parser.add_argument('--no_augmentation', action='store_true', default=False,
                    help='')
parser.add_argument('--mode', type=str, default='l1', required=False,
                    help='l1/l2/bce')
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
from cvaegan import CVAEGAN
from skimage.io import imsave
from tqdm import tqdm

use_data_augmentation = not args.no_augmentation

BS = args.batch_size
EPOCHS = args.epochs
w, h, c = args.width, args.height, args.channels
latent_dim = args.z_dim
D_ITER = 5

train_generator = data_generator(args.dataset, height=h, width=w, channel=c, shuffle=True, normalize=not use_data_augmentation, save_tags=True)
N_CLASS = len(train_generator.tags)
print('This dataset has %d unique tags'%N_CLASS)

seq = get_imgaug()

if not os.path.exists('./preview'):
    os.makedirs('./preview')

trainer = CVAEGAN(input_shape=(h, w, c), num_attrs=N_CLASS, z_dims=latent_dim, reconstruct_loss=args.mode)  
generator = trainer.return_models()[1]

i_counter = 0
for epoch in range(EPOCHS):
    print("Epoch: %d / %d"%(epoch+1, EPOCHS))
    train_generator.random_shuffle()
    with tqdm(total=len(train_generator)) as t:
        for i in range(len(train_generator)):
            image_batch, image_label = train_generator.__getitem__(i)
            if use_data_augmentation:
                image_batch = seq.augment_images(image_batch)
                image_batch = (image_batch.astype(np.float32) - 127.5) / 127.5
            
            losses = trainer.train_on_batch((image_batch, image_label))
            
            if i_counter % args.preview_iteration == 0:
                generate_images_cvaegan(generator, './preview', h, w, c, latent_dim, 5, N_CLASS, i_counter)
                trainer.save_models('./weights', i_counter)
            i_counter += 1
            
            msg = 'g_loss: {:.2f}, d_loss: {:.2f}, c_loss: {:.2f}, e_loss: {:.2f}'.format(losses['g_loss'], losses['d_loss'], losses['c_loss'], losses['e_loss'])
            t.set_description(msg)
            t.update()
            
    trainer.save_models('./weights_epoch', epoch)
