import numpy as np
import keras
from keras.models import Model
from keras.layers.merge import _Merge
from keras.optimizers import Adam
from keras.layers import Input, Add, Activation, Dense, Reshape, Flatten, LeakyReLU, GlobalAveragePooling2D, BatchNormalization
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras import metrics
from keras import backend as K
from pixel_shuffler import PixelShuffler # PixelShuffler layer
import tensorflow as tf
from functools import partial

def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.
    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def __init__(self, batch_size=8):
        super(RandomWeightedAverage, self).__init__()
        self.batch_size = batch_size

    def _merge_function(self, inputs):
        weights = K.random_uniform((self.batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def sampling(args, latent_dim=2, epsilon_std=1.0):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                          mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

def conv(f, k=3, stride=1, act=None, pad='same'):
    return Conv2D(f, (k, k), strides=(stride,stride), activation=act, kernel_initializer='he_normal', padding=pad)

def _incept_conv(f, stride=1, chs=[0.15, 0.6, 0.25]):
    def block(inputs):
        fs = [] # determine channel number
        for k in chs:
            t = max(int(k*f), 1) # at least 1 channel
            fs.append(t)

        fs[1] += f-np.sum(fs) # reminding channels allocate to 3x3 conv

        c1x1 = conv(fs[0], 1, stride=stride, pad='same') (inputs)
        c3x3 = conv(max(1, fs[1]//2), 1, stride=stride, pad='same') (inputs)
        c3x3 = LeakyReLU(0.1) (c3x3)
        c5x5 = conv(max(1, fs[2]//4), 1, stride=stride, pad='same') (inputs)
        c5x5 = LeakyReLU(0.1) (c5x5)

        c3x3 = conv(fs[1], 3, act=None, pad='same') (c3x3)
        c5x5 = conv(fs[2], 5, act=None, pad='same') (c5x5)

        output = concatenate([c1x1, c3x3, c5x5], axis=-1)
        return output
    return block

def _res_conv(f, stride=1, dropout=0.1, bn=False): # very simple residual module
    def block(inputs):
        channels = int(inputs.shape[-1])
        cs = _incept_conv(f, stride=stride) (inputs)

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

def up_lambda(inputs):
    input_shape = K.int_shape(inputs)[-3:-1] # h, w
    return tf.image.resize_bilinear(inputs, [input_shape[0]*2, input_shape[1]*2]) 

def up():
    def block(inputs):
        return Lambda(up_lambda) (inputs)
    return block

def inception_residual_discriminator(h=128, w=128, c=3, dropout_rate=0.1):

    inputs = Input(shape=(h,w,c))

    # block 1:
    b1_c1 = _res_conv(32, 1, dropout_rate) (inputs)
    b1_c2 = _res_conv(32, 2, dropout_rate, bn=True) (b1_c1)

    # block 2:
    b2_c1 = _res_conv(64, 1, dropout_rate, bn=True) (b1_c2)
    b2_c2 = _res_conv(64, 2, dropout_rate, bn=True) (b2_c1)

    # block 3:
    b3_c1 = _res_conv(128, 1, dropout_rate, bn=True) (b2_c2)
    b3_c2 = _res_conv(128, 2, dropout_rate) (b3_c1)
    
    hidden = GlobalAveragePooling2D() (b3_c2)
    dis = Dense(1, kernel_initializer='he_normal') (hidden) # We don't need 'sigmoid' here!!
    model = Model([inputs], [dis])
    return model

def inception_residual_encoder(h=128, w=128, c=3, latent_dim=2, epsilon_std=1.0, dropout_rate=0.1):

    inputs = Input(shape=(h,w,c))

    # block 1:
    b1_c1 = _res_conv(32, 1, dropout_rate) (inputs)
    b1_c2 = _res_conv(32, 2, dropout_rate, bn=True) (b1_c1)

    # block 2:
    b2_c1 = _res_conv(64, 1, dropout_rate, bn=True) (b1_c2)
    b2_c2 = _res_conv(64, 2, dropout_rate, bn=True) (b2_c1)

    # block 3:
    b3_c1 = _res_conv(128, 1, dropout_rate, bn=True) (b2_c2)
    b3_c2 = _res_conv(128, 2, dropout_rate) (b3_c1)
    
    hidden = GlobalAveragePooling2D() (b3_c2)

    z_mean =    Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)

    z = Lambda(sampling, output_shape=(latent_dim,), arguments={'latent_dim':latent_dim, 'epsilon_std':epsilon_std}) ([z_mean, z_log_var])
    model = Model([inputs], [z, z_mean, z_log_var])
    return model, b3_c2.shape

def inception_residual_decoder(original_dim, c=3, latent_dim=2, dropout_rate=0.1):

    inputs_ = Input(shape=(latent_dim,))
    transform = Dense(int(np.prod(np.asarray(original_dim)))) (inputs_)
    transform = LeakyReLU(0.1) (transform)
    reshape = Reshape(list(map(int, original_dim))) (transform)

    # block 4:
    b4_c1 = _res_conv(256, 1, dropout_rate, bn=True) (reshape)

    # block 5:
    b5_u1 = PixelShuffler() (b4_c1)
    b5_c1 = _res_conv(128, 1, dropout_rate, bn=True) (b5_u1)
    b5_c2 = _res_conv(128, 1, dropout_rate, bn=True) (b5_c1)

    # block 6:
    b6_u1 = PixelShuffler() (b5_c2)
    b6_c1 = _res_conv(64, 1, dropout_rate, bn=True) (b6_u1)
    b6_c2 = _res_conv(64, 1, dropout_rate, bn=True) (b6_c1)

    # block 7:
    b7_u1 = PixelShuffler() (b6_c2)
    b7_c1 = _res_conv(32, 1, dropout_rate, bn=True) (b7_u1)
    b7_c2 = _res_conv(32, 1, dropout_rate) (b7_c1)

    outputs = Conv2D(c, (1, 1), padding='valid', activation='tanh') (b7_c2)

    model = Model([inputs_], [outputs])
    return model

def build_inception_residual_vae(h=128, w=128, c=3, latent_dim=2, epsilon_std=1.0, dropout_rate=0.1):

    images = Input(shape=(h,w,c), name='vae_input_images')

    encoder_model, shape = inception_residual_encoder(h=h, w=w, c=c, latent_dim=latent_dim, epsilon_std=epsilon_std, dropout_rate=dropout_rate)
    z, z_mean, z_log_var = encoder_model(images)
    decoder_model = inception_residual_decoder(shape[1:], c=c, latent_dim=latent_dim, dropout_rate=dropout_rate)
    outputs = decoder_model(z)
    
    d_loss  = K.mean( 0.5 * K.sum(K.square(images - outputs), axis=-1) / (2*(epsilon_std**2)) + np.log(epsilon_std)) # gaussian_log_likelihood, may makes outputs blurry
    kl_loss = K.mean(-0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)) # KL-divergence

    vae = Model([images], [outputs])
    vae.add_loss(d_loss) # add reconstruction loss (gaussian_log_likelihood) to decoder(generator)
    vae.add_loss(kl_loss,  inputs=[encoder_model]) # add KL-divergence loss to encoder
    vae.compile(optimizer='rmsprop', loss=None)

    return vae, encoder_model, decoder_model

def build_vae_gan(h=128, w=128, c=3, latent_dim=2, epsilon_std=1.0, dropout_rate=0.1, GRADIENT_PENALTY_WEIGHT=10, batch_size=8, use_vae=False, vae_use_sse=True):
    
    vae_input = Input(shape=(h,w,c))
    encoder_model, shape = inception_residual_encoder(h=h, w=w, c=c, latent_dim=latent_dim, epsilon_std=epsilon_std, dropout_rate=dropout_rate)
    z, z_mean, z_log_var = encoder_model(vae_input)
    generator = inception_residual_decoder(shape[1:], c=c, latent_dim=latent_dim, dropout_rate=dropout_rate)
    
    discriminator = inception_residual_discriminator(h=h,w=w,c=c,dropout_rate=dropout_rate)
    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False
    
    generator_input = Input(shape=(latent_dim,))
    generator_layers = generator(generator_input)
    
    discriminator_layers_for_generator = discriminator(generator_layers)
    generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
    generator_model.add_loss(K.mean(discriminator_layers_for_generator))
    # We use the Adam paramaters from Gulrajani et al.
    generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=None)
    
    if use_vae:
        vae_output = generator(z)
        d_loss  = K.mean( 0.5 * K.sum(K.square(vae_input - vae_output), axis=-1) / (2*(epsilon_std**2)) + np.log(epsilon_std)) # gaussian_log_likelihood
        kl_loss = K.mean(-0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)) # KL-divergence
        discriminator_layers_for_vae = discriminator(vae_output)
        vae_model = Model(inputs=[vae_input], outputs=[discriminator_layers_for_vae])
        vae_model.add_loss(kl_loss, inputs=[encoder_model])
        vae_model.add_loss(K.mean(discriminator_layers_for_vae))
        if vae_use_sse:
            vae_model.add_loss(d_loss) # may makes outputs blurry
        vae_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=None)

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

    # We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
    averaged_samples = RandomWeightedAverage(batch_size)([real_samples, generated_samples_for_discriminator])
    # We then run these samples through the discriminator as well. Note that we never really use the discriminator
    # output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
    averaged_samples_out = discriminator(averaged_samples)

    # The gradient penalty loss function requires the input averaged samples to get gradients. However,
    # Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
    # of the function with the averaged samples here.

    discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                                outputs=[discriminator_output_from_real_samples,
                                         discriminator_output_from_generator,
                                         averaged_samples_out])
    discriminator_model.add_loss(K.mean(discriminator_output_from_real_samples) + K.mean(-discriminator_output_from_generator)
                                                                                + gradient_penalty_loss(averaged_samples_out, averaged_samples, GRADIENT_PENALTY_WEIGHT))
    discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=None)

    return (generator_model, discriminator_model, vae_model, encoder_model, generator, discriminator) if use_vae else (generator_model, discriminator_model, generator, discriminator)

if __name__ == '__main__':
    vae, encoder, decoder = build_inception_residual_vae(h=32, w=32, c=1, dropout_rate=0.2)
    vae.summary()
    encoder.summary()
    decoder.summary()