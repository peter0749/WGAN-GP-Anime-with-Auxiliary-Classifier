import sys
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

parser = argparse.ArgumentParser(description='Music Generation with VAE')
parser.add_argument('output', metavar='output', type=str,

                    help='')
parser.add_argument('--model', type=str, default='./decoder.h5', required=False,

                    help='model')
parser.add_argument('--x', type=float, default=0.0, required=False,

                    help='x')
parser.add_argument('--y', type=float, default=0.0, required=False,

                    help='y')
parser.add_argument('--threshold', type=float, default=0.5, required=False,

                    help='threshold')
args = parser.parse_args()
x, y = args.x, args.y

model = load_model(args.model, custom_objects={'PixelShuffler':PixelShuffler})
m = (np.squeeze(model.predict(np.array([[x, y]]), batch_size=1)) * 127.5 + 127.5).astype(np.uint8)
imsave(args.output, m)
