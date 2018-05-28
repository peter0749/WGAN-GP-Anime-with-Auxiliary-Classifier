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
from models import up_bilinear
from skimage.io import imsave
from pixel_shuffler import PixelShuffler
from tools import generate_image_interpolation
import argparse

parser = argparse.ArgumentParser(description='Image Generation with VAE/GAN')
parser.add_argument('output', metavar='output', type=str, help='')
parser.add_argument('--model', type=str, default='./decoder.h5', required=False, help='model')
parser.add_argument('--n' , type=int, default=9, required=False, help='Interpolation points')
parser.add_argument('--dt', type=int, default=12, required=False, help='Interpolation steps')
parser.add_argument('--nr', type=int, default=7, required=False, help='rows')
parser.add_argument('--nc', type=int, default=7, required=False, help='cols')
parser.add_argument('--std', type=float, default=0.6, required=False, help='')
parser.add_argument('--batch_size', type=int, default=8, required=False, help='')
args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)

model = load_model(args.model, custom_objects={'tf':tf, 'PixelShuffler':PixelShuffler, 'up_bilinear':up_bilinear})
generate_image_interpolation(model, args.output, *model.output_shape[-3:], model.input_shape[-1], args.std, args.nr, args.nc, args.dt, args.n, batch_size=args.batch_size)
