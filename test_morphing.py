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
from tools import back_to_z, z_interpolation
from skimage.io import imsave, imread
from skimage.transform import resize
from pixel_shuffler import PixelShuffler
import cv2
import argparse

parser = argparse.ArgumentParser(description='Image Generation with GAN')
parser.add_argument('input_1', metavar='input_1', type=str, help='')
parser.add_argument('input_2', metavar='input_2', type=str, help='')
parser.add_argument('output', metavar='output', type=str, help='')
parser.add_argument('--decoder', type=str, default='./decoder.h5', required=False, help='decoder')
parser.add_argument('--encoder', type=str, default='./encoder.h5', required=False, help='encoder')
parser.add_argument('--std', type=float, default=1.0, required=False, help='')
parser.add_argument('--batch_size', type=int, default=8, required=False, help='')
parser.add_argument('--iterations', type=int, default=500, required=False, help='')
parser.add_argument('--sample_n', type=int, default=16, required=False, help='')
parser.add_argument('--interpolation_method', type=str, default='bilinear', required=False, help='bilinear / bicubic')
parser.add_argument('--color_morphing', action='store_true', default=False, help='')
args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)
    
CV_INTER = cv2.INTER_LINEAR if args.interpolation_method=='bilinear' else cv2.INTER_CUBIC

decoder = load_model(args.decoder, custom_objects={'tf':tf, 'PixelShuffler':PixelShuffler, 'up_bilinear':up_bilinear})
encoder = load_model(args.encoder, custom_objects={'tf':tf, 'PixelShuffler':PixelShuffler, 'up_bilinear':up_bilinear}) if os.path.exists(args.encoder) else None
img_1 = resize(imread(args.input_1), decoder.output_shape[-3:-1], preserve_range=True, order=1).astype(np.float32)
img_2 = resize(imread(args.input_2), decoder.output_shape[-3:-1], preserve_range=True, order=1).astype(np.float32)
z_encoder = back_to_z(decoder, encoder)
z_1 = z_encoder.get_z((img_1-127.5)/127.5, args.std, iterations=args.iterations, return_img=False)
z_2 = z_encoder.get_z((img_2-127.5)/127.5, args.std, iterations=args.iterations, return_img=False)
zs = z_interpolation([z_1, z_2], n=args.sample_n)
y_h = np.clip(decoder.predict(zs, batch_size=args.batch_size) * 127.5 + 127.5, 0, 255)
imgs_forward = [img_1]
imgs_backward= [img_2]
img_t = np.clip(img_1, 0, 255)
for t in range(1, len(y_h)):
    y_t_diff = y_h[t] - y_h[t-1] if args.color_morphing else 0
    prevFrm = cv2.cvtColor(y_h[t-1], cv2.COLOR_RGB2YCrCb)[...,0:1]
    currFrm = cv2.cvtColor(y_h[t  ], cv2.COLOR_RGB2YCrCb)[...,0:1]
    y_t_flow = cv2.calcOpticalFlowFarneback(currFrm, prevFrm, None, 0.5, 3, 15, 3, 5, 1.2, 0).astype(np.float32)
    height, width = y_t_flow.shape[:2]
    y_t_flow[...,0] += np.arange(width)
    y_t_flow[...,1] += np.arange(height)[...,np.newaxis]
    img_t = np.clip(cv2.remap(img_t, y_t_flow, None, CV_INTER) + y_t_diff, 0, 255)
    imgs_forward.append(img_t)
    # imsave(args.output+'/t_%d.png'%t, np.round(np.clip(np.concatenate([y_h[t], img_t, img_1, img_2], axis=1), 0, 255)).astype(np.uint8))
img_t = np.clip(img_2, 0, 255)
for t in reversed(range(0, len(y_h)-1)):
    y_t_diff = y_h[t] - y_h[t+1] if args.color_morphing else 0
    prevFrm = cv2.cvtColor(y_h[t+1], cv2.COLOR_RGB2YCrCb)[...,0:1]
    currFrm = cv2.cvtColor(y_h[t  ], cv2.COLOR_RGB2YCrCb)[...,0:1]
    y_t_flow = cv2.calcOpticalFlowFarneback(currFrm, prevFrm, None, 0.5, 3, 15, 3, 5, 1.2, 0).astype(np.float32)
    height, width = y_t_flow.shape[:2]
    y_t_flow[...,0] += np.arange(width)
    y_t_flow[...,1] += np.arange(height)[...,np.newaxis]
    img_t = np.clip(cv2.remap(img_t, y_t_flow, None, CV_INTER) + y_t_diff, 0, 255)
    imgs_backward.append(img_t)
    # imsave(args.output+'/t_%d.png'%t, np.round(np.clip(np.concatenate([y_h[t], img_t, img_1, img_2], axis=1), 0, 255)).astype(np.uint8))
assert len(imgs_forward)==len(imgs_backward)
for t, img_f, img_b in zip(range(len(imgs_forward)), imgs_forward, reversed(imgs_backward)):
    img_t = (img_f*(len(imgs_forward)-1-t) + img_b*t) / (len(imgs_forward)-1)
    imsave(args.output+'/t_%d.png'%t, np.round(np.clip(np.concatenate([y_h[t], img_t, img_1, img_2], axis=1), 0, 255)).astype(np.uint8))