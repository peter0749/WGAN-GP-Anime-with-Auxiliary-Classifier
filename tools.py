import os
import numpy as np
from skimage.io import imread
import glob
from keras.utils import Sequence
from keras.callbacks import Callback
from skimage.io import imsave
from skimage.transform import resize
from skimage.color import gray2rgb
from keras.datasets import mnist

class mnist_generator(Sequence):
    def __init__(self, images, height=32, width=32, batch_size=8):
        self.bs = batch_size
        self.imgs = images
        self.h = height
        self.w = width
    def __len__(self):
        return int(np.ceil(float(len(self.imgs))/self.bs))
    def __getitem__(self, idx):
        l_bound = idx     * self.bs
        r_bound = (idx+1) * self.bs
        if r_bound > len(self.imgs):
            r_bound = len(self.imgs)
            l_bound = r_bound - self.bs
        x_batch = np.zeros((r_bound - l_bound, self.h, self.w, 1))
        for n, img in enumerate(self.imgs[l_bound:r_bound]):
            img = resize(np.squeeze(img), (self.h, self.w), order=1, preserve_range=True)[...,np.newaxis]
            x_batch[n] = np.clip((img.astype(np.float32)-127.5) / 127.5, -1, 1)
        return x_batch, None

class data_generator(Sequence):
    def __init__(self, images_path, height=128, width=128, channel=3, batch_size=8, shuffle=True):
        self.bs = batch_size
        self.imgs = glob.glob(images_path+'/**/*.jpg') ## paths
        self.imgs.extend(glob.glob(images_path+'/**/*.png'))
        self.h = height
        self.w = width
        self.c = channel
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.imgs)
    def __len__(self):
        return int(np.ceil(float(len(self.imgs))/self.bs))
    def random_shuffle(self):
        if self.shuffle:
            np.random.shuffle(self.imgs)
    def __getitem__(self, idx):
        l_bound = idx     * self.bs
        r_bound = (idx+1) * self.bs
        if r_bound > len(self.imgs):
            r_bound = len(self.imgs)
            l_bound = r_bound - self.bs
        x_batch = np.zeros((r_bound - l_bound, self.h, self.w, self.c))
        for n, imgp in enumerate(self.imgs[l_bound:r_bound]):
            img = imread(str(imgp), as_grey=(self.c==1))
            if img.shape[0]!=self.h or img.shape[1]!=self.w:
                img = resize(img, (self.h, self.w), order=2, preserve_range=True)
            if img.ndim==2:
                img = np.expand_dims(img, -1)
            if self.c == 3:
                img = gray2rgb(img)
            img = img[...,:self.c]
            x_batch[n] = np.clip((img.astype(np.float32)-127.5) / 127.5, -1, 1)
        return x_batch, None

def generate_images(generator, path, h, w, c, std, nr, nc, epoch, batch_size=1):
    grid_x = np.linspace(-15, 15, nc)
    grid_y = np.linspace(-15, 15, nr)
    figure = np.zeros((h * nr, w * nc, c))
    for ri, y in enumerate(grid_y):
        for ci, x in enumerate(grid_x):
            z_sample = np.array([[x, y]]) * std
            figure[h*ri:h*(ri+1), w*ci:w*(ci+1)] = generator.predict(z_sample, batch_size=1, verbose=0)[0]
    figure = np.squeeze(np.clip(figure * 127.5 + 127.5, 0, 255).astype(np.uint8))
    imsave(os.path.join(path, 'epoch_{:02d}.jpg'.format(epoch)), figure)
    generator.save(os.path.join(path, 'weights_{:02d}.h5'.format(epoch)))

class Preview(Callback):
    def __init__(self, decoder, path, h, w, c=3, nr=15, nc=15, std=1.0, batch_size=1, save_weights=True):
        self.nr = nr
        self.nc = nc
        self.path = path
        self.decoder = decoder
        self.batch_size = batch_size
        self.sv = save_weights
        self.h, self.w, self.c = h, w, c
        self.std = std
    def on_epoch_end(self, epoch, logs):
        nr, nc = self.nr, self.nc
        h, w, c = self.h, self.w, self.c
        path = self.path
        generate_images(self.decoder, self.path, self.h, self.w, self.c, self.std, self.nr, self.nc, epoch, self.batch_size)
