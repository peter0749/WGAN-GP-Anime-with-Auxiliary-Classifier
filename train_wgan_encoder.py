import argparse
parser = argparse.ArgumentParser(description='WGAN-GP')
parser.add_argument('--dataset', type=str, required=True,
                    help='path to dataset')
parser.add_argument('--decoder', type=str, required=True,
                    help='path to decoder')
parser.add_argument('--batch_size', type=int, default=32, required=False,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=1000, required=False,
                    help='epochs')
parser.add_argument('--preview_iteration', type=int, default=1000, required=False,
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
from models import make_encoder, up_bilinear
from pixel_shuffler import PixelShuffler
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from skimage.io import imsave
from tqdm import tqdm

use_data_augmentation = not args.no_augmentation

decoder = load_model(args.decoder, custom_objects={'tf':tf, 'PixelShuffler':PixelShuffler, 'up_bilinear':up_bilinear})
for l in decoder.layers:
    l.trainable = False
decoder.trainable = False

BS = args.batch_size
EPOCHS = args.epochs
h, w, c = decoder.output_shape[-3:]
latent_dim = decoder.input_shape[-1]

train_generator = data_generator(args.dataset, height=h, width=w, channel=c, shuffle=True, normalize=not use_data_augmentation, save_tags=False)

encoder_model, encoder = make_encoder(decoder)

seq = get_imgaug()

if not os.path.exists('./preview'):
    os.makedirs('./preview')
    
def make_some_noise():
    return np.random.normal(0, args.std, (BS, latent_dim)).astype(np.float32)

i_counter = 0
AE = 0
for epoch in range(EPOCHS):
    print("Epoch: %d / %d"%(epoch+1, EPOCHS))
    train_generator.random_shuffle()
    with tqdm(total=len(train_generator)) as t:
        for i in range(len(train_generator)):
            image_batch, _ = train_generator.__getitem__(i)
            if use_data_augmentation:
                image_batch = seq.augment_images(image_batch)
                image_batch = (image_batch.astype(np.float32) - 127.5) / 127.5
            
            z = make_some_noise()
            AE += np.mean(encoder_model.train_on_batch([image_batch, z], None))
            
            if i_counter % args.preview_iteration == 0:
                img = decoder.predict(encoder.predict(image_batch[0:1]))[0]
                img = np.append(img, image_batch[0], axis=1)
                imsave('./preview/ite_{:d}.jpg'.format(i_counter), img)
            i_counter += 1
            
            msg = 'loss: {:.2f}'.format(AE / i_counter)
            t.set_description(msg)
            t.update()
            
    encoder.save('./encoder.h5')
