import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import keras
from keras.utils import to_categorical
from keras import backend as K
K.set_session(session)
from keras.models import load_model
from models import up_bilinear
from skimage.io import imsave
from pixel_shuffler import PixelShuffler
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Image Generation with GAN')
parser.add_argument('output', metavar='output', type=str, help='')
parser.add_argument('label',  metavar='label',  type=str, help='')
parser.add_argument('tag_file',  metavar='tag_file',  type=str, help='')
parser.add_argument('--model', type=str, default='./decoder.h5', required=False, help='model')
parser.add_argument('--n', type=int, default=64, required=False, help='')
parser.add_argument('--std', type=float, default=0.7, required=False, help='')
parser.add_argument('--batch_size', type=int, default=8, required=False, help='')
args = parser.parse_args()

tags = pd.read_csv(args.tag_file) # order is manner
mask = (tags['tags']==args.label)
assert mask.any(), '%s is not in \'%s\'!'%(args.label, args.tag_file)
label_idx = np.argmax(mask)

if not os.path.exists(args.output):
    os.makedirs(args.output)

model = load_model(args.model, custom_objects={'tf':tf, 'PixelShuffler':PixelShuffler, 'up_bilinear':up_bilinear})
N_CLASS = len(tags['tags'])
noise  = np.random.normal(0, args.std, (args.n, model.input_shape[-1]-N_CLASS))
labels = to_categorical(np.asarray([label_idx]*args.n), N_CLASS)
z = np.append(noise, labels, axis=-1)

m = (model.predict(z, batch_size=args.batch_size) * 127.5 + 127.5).astype(np.uint8)

for i in range(args.n):
    imsave(os.path.join(args.output, 'output_{:04d}.png'.format(i)), np.squeeze(m[i]))
