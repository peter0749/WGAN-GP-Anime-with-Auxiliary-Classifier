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
from vae_model import up_bilinear
from tools import back_to_z
from skimage.io import imsave, imread
from skimage.transform import resize
from pixel_shuffler import PixelShuffler
import argparse

parser = argparse.ArgumentParser(description='Image Generation with VAE/GAN')
parser.add_argument('input', metavar='input', type=str, help='')
parser.add_argument('output', metavar='output', type=str, help='')
parser.add_argument('--model', type=str, default='./decoder.h5', required=False, help='model')
parser.add_argument('--std', type=float, default=1.0, required=False, help='')
parser.add_argument('--iterations', type=int, default=500, required=False, help='')
parser.add_argument('--runs', type=int, default=10, required=False, help='')
args = parser.parse_args()

model = load_model(args.model, custom_objects={'tf':tf, 'PixelShuffler':PixelShuffler, 'up_bilinear':up_bilinear})
for i in range(args.runs):
    print('Runs: %d / %d'%(i+1, args.runs))
    img = (resize(imread(args.input), model.output_shape[-3:-1], preserve_range=True).astype(np.float32) - 127.5) / 127.5
    z, img_reconstruct = back_to_z(img, model, args.std, iterations=args.iterations, return_img=True)
    output_img = np.round(np.concatenate((np.squeeze(img), np.squeeze(img_reconstruct)), axis=1) * 127.5 + 127.5).astype(np.uint8)
    filename, ext = os.path.splitext(args.output)
    imsave(filename+'_%d'+ext, output_img)
