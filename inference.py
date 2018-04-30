import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import keras
from keras import backend as K
K.set_session(session)
from keras.models import *
from pixel_shuffler import PixelShuffler
from vae_model import build_residual_vae
from skimage.io import imsave
import argparse

parser = argparse.ArgumentParser(description='Image Generation with VAE/GAN')
parser.add_argument('output', metavar='output', type=str, help='')
parser.add_argument('--model', type=str, default='./decoder.h5', required=False, help='model')
parser.add_argument('--n', type=int, default=64, required=False, help='')
parser.add_argument('--d', type=int, default=100, required=False, help='')
parser.add_argument('--std', type=float, default=1.0, required=False, help='')
parser.add_argument('--batch_size', type=int, default=8, required=False, help='')
args = parser.parse_args()

model = load_model(args.model, custom_objects={'PixelShuffler':PixelShuffler})
m = (model.predict(np.random.normal(0, args.std, (args.n, args.d)), batch_size=args.batch_size) * 127.5 + 127.5).astype(np.uint8)

for i in range(args.n):
    imsave(os.path.join(args.output, 'output_{:04d}'.format(i)), np.squeeze(m[i]))
