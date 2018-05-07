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
from vae_model import build_residual_vae
from skimage.io import imsave, imread
from skimage.color import gray2rgb
from glob import glob
from pixel_shuffler import PixelShuffler
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Image Generation with VAE/GAN')
parser.add_argument('input', metavar='input_dir', type=str, help='input_dir')
parser.add_argument('output', metavar='output_dir', type=str, help='output_dir')
parser.add_argument('--model', type=str, default='./generator_A.h5', required=False, help='model')
args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)

model = load_model(args.model, custom_objects={'PixelShuffler':PixelShuffler})

paths = glob(args.input+'/*.jpg')
paths.extend(glob(args.input+'/*.png'))
for img_path in tqdm(paths, total=len(paths)):
    img = (gray2rgb(imread(img_path, as_grey=False))[...,:3] - 127.5) / 127.5
    o_path = os.path.join(args.output, os.path.split(img_path)[-1])
    img_out = np.round(model.predict(img[np.newaxis,...], batch_size=1, verbose=0)[0] * 127.5 + 127.5).astype(np.uint8)
    imsave(o_path, img_out)
