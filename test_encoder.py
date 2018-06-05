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
from skimage.io import imsave, imread
from pixel_shuffler import PixelShuffler
from skimage.transform import resize
import argparse

parser = argparse.ArgumentParser(description='Image Generation with GAN')
parser.add_argument('input_img', metavar='input_img', type=str, help='')
parser.add_argument('output_img', metavar='output_img', type=str, help='')
parser.add_argument('--decoder', type=str, default='./decoder.h5', required=False, help='model')
parser.add_argument('--encoder', type=str, default='./encoder.h5', required=False, help='model')
args = parser.parse_args()

decoder = load_model(args.decoder, custom_objects={'tf':tf, 'PixelShuffler':PixelShuffler, 'up_bilinear':up_bilinear})
encoder = load_model(args.encoder, custom_objects={'tf':tf, 'PixelShuffler':PixelShuffler, 'up_bilinear':up_bilinear})
query_img = (resize(imread(args.input_img), encoder.input_shape[-3:], preserve_range=True).astype(np.float32) - 127.5) / 127.5
output_img = (decoder.predict(encoder.predict(query_img[np.newaxis,...]))[0] * 127.5 + 127.5).astype(np.uint8)
imsave(args.output_img, output_img)
