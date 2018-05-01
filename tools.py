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
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

def get_imgaug():
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    
    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                rotate=(-3, 3), # rotate by -45 to +45 degrees
                shear=(-3, 3), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode='constant' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 2),
                [
                    iaa.Add((-5, 5), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-5, 5)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.Multiply((0.9, 1.1), per_channel=0.5)
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    return seq

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

def get_all_data(images_path, height=128, width=128):
    img_path_list = glob.glob(images_path+'/**/*.jpg') ## paths
    img_path_list.extend(glob.glob(images_path+'/**/*.png'))
    def read_img(img_name):
        img = imread(img_name, as_grey=False)
        if img.ndim==2:
            img = gray2rgb(img)
        if img.shape[0]!=height or img.shape[1]!=width:
            order = 2 if img.shape[0]<height or img.shape[1]<width else 0 # reduce artifact
            img = resize(img, (height, width), order=order, preserve_range=True)
        return img[...,:3] # discard alpha channel (if exist)
    images = np.asarray([ read_img(imgp) for imgp in img_path_list], dtype=np.uint8) # save memory space
    return images
    

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
                order = 2 if img.shape[0]<self.h or img.shape[1]<self.w else 0
                img = resize(img, (self.h, self.w), order=order, preserve_range=True)
            if img.ndim==2:
                img = np.expand_dims(img, -1)
            if self.c == 3:
                img = gray2rgb(img)
            img = img[...,:self.c]
            x_batch[n] = np.clip((img.astype(np.float32)-127.5) / 127.5, -1, 1)
        return x_batch, None

def generate_images(generator, path, h, w, c, latent_dim, std, nr, nc, epoch, batch_size=1):
    noise = np.random.normal(0, std, (nr*nc, latent_dim))
    generated = generator.predict(noise, batch_size=batch_size, verbose=0)
    figure = np.zeros((h * nr, w * nc, c))
    for ri in range(nr):
        for ci in range(nc):
            figure[h*ri:h*(ri+1), w*ci:w*(ci+1)] = generated[ri*nc+ci]
    figure = np.squeeze(np.clip(figure * 127.5 + 127.5, 0, 255).astype(np.uint8))
    imsave(os.path.join(path, 'epoch_{:02d}.jpg'.format(epoch)), figure)
    generator.save(os.path.join(path, 'weights_{:02d}.h5'.format(epoch)))

class Preview(Callback):
    def __init__(self, decoder, path, h, w, c=3, latent_dim=2, nr=15, nc=15, std=1.0, batch_size=1, save_weights=True):
        self.nr = nr
        self.nc = nc
        self.path = path
        self.decoder = decoder
        self.batch_size = batch_size
        self.sv = save_weights
        self.h, self.w, self.c = h, w, c
        self.std = std
        self.latent_dim = latent_dim
    def on_epoch_end(self, epoch, logs):
        nr, nc = self.nr, self.nc
        h, w, c = self.h, self.w, self.c
        path = self.path
        generate_images(self.decoder, self.path, self.h, self.w, self.c, self.latent_dim, self.std, self.nr, self.nc, epoch, self.batch_size)

        
