import numpy as np
import keras
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.layers.merge import _Merge
from keras.layers import Input, Add, Activation, Dense, Reshape, Flatten, GlobalAveragePooling2D, BatchNormalization, LeakyReLU
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.merge import concatenate
from keras import metrics
from keras import backend as K
from pixel_shuffler import PixelShuffler # PixelShuffler layer
import tensorflow as tf
from functools import partial

def sampling(args, latent_dim=2, epsilon_std=1.0):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                          mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

def conv(f, k=3, stride=1, act=None, pad='same'):
    return Conv2D(f, (k, k), strides=(stride,stride), activation=act, kernel_initializer='he_normal', padding=pad)

def _res_conv(f, stride=1, dropout=0.1, bn=False): # very simple residual module
    def block(inputs):
        channels = int(inputs.shape[-1])
        cs = conv(f, 3, stride=stride) (inputs)

        if f!=channels or stride!=1:
            t1 = conv(f, 1, stride=stride, act=None, pad='same') (inputs) # identity mapping
        else:
            t1 = inputs

        out = Add()([t1, cs]) # t1 + c2
        if bn:
            out = BatchNormalization() (out)
        out = LeakyReLU(0.1) (out)
        if dropout>0:
            out = Dropout(dropout) (out)
        return out
    return block
    

def up(filters, dropout_rate=0, bn=False):
    def block(inputs):
        x = conv(filters * 4, 3, 1) (inputs)
        if bn:
            x = BatchNormalization() (x)
        x = LeakyReLU(0.2) (x)
        if dropout_rate>0:
            x = Dropout(dropout_rate) (x)
        x = PixelShuffler()(x)
        return x
    return block

def residual_discriminator(h=128, w=128, c=3, dropout_rate=0.1):

    inputs = Input(shape=(h,w,c)) # 48x48@c

    # block 1:
    x = conv(32, 3, 2, pad='same') (inputs) # 24x24@32
    x = BatchNormalization() (x)
    x = LeakyReLU(0.2) (x)
    x = Dropout(dropout_rate) (x)
    
    # block 2:
    x = conv(64, 3, 2, pad='same') (x) # 12x12@64
    x = BatchNormalization() (x)
    x = LeakyReLU(0.2) (x)
    x = Dropout(dropout_rate) (x)
    x = _res_conv(64, 1, dropout_rate, True) (x)
    
    # block 3:
    x = conv(128, 3, 2) (x) # 6x6@128
    x = BatchNormalization() (x)
    x = LeakyReLU(0.2) (x)
    x = Dropout(dropout_rate) (x)
    x = _res_conv(128, 1, dropout_rate, True) (x)
    
    p = x # 6x6@128
    
    hidden = Flatten() (x) # 6*6*256
    dis = Dense(1, kernel_initializer='he_normal') (hidden) # We don't need 'sigmoid' here!!
    model = Model([inputs], [dis])
    return model

def residual_encoder(h=128, w=128, c=3, latent_dim=2, epsilon_std=1.0, dropout_rate=0.1):

    inputs = Input(shape=(h,w,c)) # 48x48@c

    # block 1:
    x = conv(32, 3, 2, pad='same') (inputs) # 24x24@32
    x = BatchNormalization() (x)
    x = LeakyReLU(0.2) (x)
    x = Dropout(dropout_rate) (x)
    
    # block 2:
    x = conv(64, 3, 2, pad='same') (x) # 12x12@64
    x = BatchNormalization() (x)
    x = LeakyReLU(0.2) (x)
    x = Dropout(dropout_rate) (x)
    x = _res_conv(64, 1, dropout_rate, True) (x)
    
    # block 3:
    x = conv(128, 3, 2) (x) # 6x6@128
    x = BatchNormalization() (x)
    x = LeakyReLU(0.2) (x)
    x = Dropout(dropout_rate) (x)
    x = _res_conv(128, 1, dropout_rate, True) (x)
    
    p = x # 6x6@128
    
    hidden = Flatten() (x)

    z_mean =    Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)

    z = Lambda(sampling, output_shape=(latent_dim,), arguments={'latent_dim':latent_dim, 'epsilon_std':epsilon_std}) ([z_mean, z_log_var])
    model = Model([inputs], [z, z_mean, z_log_var])
    return model, int(p.shape[1]), int(p.shape[2]) # h, w

def residual_decoder(h, w, c=3, latent_dim=2, dropout_rate=0.1):

    hid_channel = 128
    inputs_ = Input(shape=(latent_dim,))
    transform = Dense(h*w*hid_channel) (inputs_) # 6*6*hid_channel
    transform = BatchNormalization() (transform)
    transform = LeakyReLU(0.2) (transform)
    reshape = Reshape((h,w,hid_channel)) (transform) # 6x6@hid_channel

    x = up(128, dropout_rate, bn=True) (reshape) # 12x12@128
    x = _res_conv(128, 1, dropout_rate, True) (x)
    
    x = up(64,  dropout_rate, bn=True) (x) # 24x24@64
    x = _res_conv(64, 1, dropout_rate, True) (x)
    
    x = up(32,  dropout_rate, bn=True) (x) # 48x48@32
    x = _res_conv(32, 1, dropout_rate, True) (x)
    
    outputs = Conv2D(c, (1, 1), padding='valid', activation='tanh') (x) # 48x48@c

    model = Model([inputs_], [outputs])
    return model

def build_residual_vae(h=128, w=128, c=3, latent_dim=2, epsilon_std=1.0, dropout_rate=0.1):

    optimizer = RMSprop(lr=0.0005)
    images = Input(shape=(h,w,c), name='vae_input_images')

    encoder_model, t_h, t_w = residual_encoder(h=h, w=w, c=c, latent_dim=latent_dim, epsilon_std=epsilon_std, dropout_rate=dropout_rate)
    z, z_mean, z_log_var = encoder_model(images)
    decoder_model = residual_decoder(t_h, t_w, c=c, latent_dim=latent_dim, dropout_rate=dropout_rate)
    outputs = decoder_model(z)
    
    d_loss  = K.mean( 0.5 * K.sum(K.square(images - outputs), axis=-1) / (2*(epsilon_std**2)) + np.log(epsilon_std)) # gaussian_log_likelihood, may makes outputs blurry
    kl_loss = K.mean(-0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)) # KL-divergence

    vae = Model([images], [outputs])
    vae.add_loss(d_loss) # add reconstruction loss (gaussian_log_likelihood) to decoder(generator)
    vae.add_loss(kl_loss,  inputs=[encoder_model]) # add KL-divergence loss to encoder
    vae.compile(optimizer=optimizer, loss=None)

    return vae, encoder_model, decoder_model

def build_vae_gan(h=128, w=128, c=3, latent_dim=2, epsilon_std=1.0, dropout_rate=0.1, GRADIENT_PENALTY_WEIGHT=10, batch_size=8, use_vae=False, vae_use_sse=True):
    
    optimizer = Adam(0.0002, 0.5)
    
    vae_input = Input(shape=(h,w,c))
    encoder_model, t_h, t_w = residual_encoder(h=h, w=w, c=c, latent_dim=latent_dim, epsilon_std=epsilon_std, dropout_rate=dropout_rate)
    z, z_mean, z_log_var = encoder_model(vae_input)
    generator = residual_decoder(t_h, t_w, c=c, latent_dim=latent_dim, dropout_rate=dropout_rate)
    
    discriminator = residual_discriminator(h=h,w=w,c=c,dropout_rate=dropout_rate)
    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False
    
    generator_input = Input(shape=(latent_dim,))
    generator_layers = generator(generator_input)
    
    discriminator_layers_for_generator = discriminator(generator_layers)
    generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
    generator_model.add_loss(0.5 * K.mean(K.square(discriminator_layers_for_generator - 0.99)))
    generator_model.compile(optimizer=optimizer, loss=None)
    
    if use_vae:
        vae_output = generator(z)
        d_loss  = K.mean( 0.5 * K.sum(K.square(vae_input - vae_output), axis=-1) / (2*(epsilon_std**2)) + np.log(epsilon_std)) # gaussian_log_likelihood
        kl_loss = K.mean(-0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)) # KL-divergence
        discriminator_layers_for_vae = discriminator(vae_output)
        vae_model = Model(inputs=[vae_input], outputs=[discriminator_layers_for_vae])
        vae_model.add_loss(kl_loss, inputs=[encoder_model])
        vae_model.add_loss(0.5 * K.mean(K.square(discriminator_layers_for_vae - 0.99)))
        if vae_use_sse:
            vae_model.add_loss(d_loss) # may makes outputs blurry
        vae_model.compile(optimizer=optimizer, loss=None)

    # Now that the generator_model is compiled, we can make the discriminator layers trainable.
    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    discriminator.trainable = True
    generator.trainable = False

    # The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
    # The noise seed is run through the generator model to get generated images. Both real and generated images
    # are then run through the discriminator. Although we could concatenate the real and generated images into a
    # single tensor, we don't (see model compilation for why).
    real_samples = Input(shape=(h, w, c))
    generator_input_for_discriminator = Input(shape=(latent_dim,))
    generated_samples_for_discriminator = generator(generator_input_for_discriminator)
    discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
    discriminator_output_from_real_samples = discriminator(real_samples)

    discriminator_real = Model([real_samples], [discriminator_output_from_real_samples])
    discriminator_fake = Model([generator_input_for_discriminator], [discriminator_output_from_generator])
    
    discriminator_real.add_loss(0.5 * K.mean(K.square(discriminator_output_from_real_samples - 0.99))) # one side soft label
    discriminator_fake.add_loss(0.5 * K.mean(K.square(discriminator_output_from_generator)))
    
    discriminator_real.compile(optimizer=optimizer, loss=None)
    discriminator_fake.compile(optimizer=optimizer, loss=None)

    return (generator_model, discriminator_real, discriminator_fake, vae_model, encoder_model, generator, discriminator) if use_vae else (generator_model, discriminator_real, discriminator_fake, generator, discriminator)

if __name__ == '__main__':
    vae, encoder, decoder = build_residual_vae(h=32, w=32, c=1, dropout_rate=0.2)
    vae.summary()
    encoder.summary()
    decoder.summary()