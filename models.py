import numpy as np
import keras
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.merge import _Merge
from keras.layers import Input, Add, Activation, Dense, Reshape, Flatten, GlobalAveragePooling2D, LeakyReLU, GaussianNoise
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.merge import concatenate
from keras.regularizers import l2
from keras import metrics
from keras import backend as K
from pixel_shuffler import PixelShuffler
import tensorflow as tf
from weightnorm import AdamWithWeightnorm

def RandomWeightedAverage():
    def block(input_list):
        input1, input2 = input_list
        weights = K.random_uniform((K.shape(input1)[0], 1, 1, 1))
        return (weights * input1) + ((1 - weights) * input2)
    return Lambda(block)

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

def sampling(args, latent_dim=2, epsilon_std=1.0):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                          mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

def conv(f, k=3, stride=1, act=None, pad='same'):
    return Conv2D(f, (k, k), strides=(stride,stride), activation=act, kernel_initializer='he_normal', padding=pad)

def _res_conv(f, k=3, dropout=0.1): # very simple residual module
    def block(inputs):
        channels = int(inputs.shape[-1])
        cs = conv(f, k, stride=1) (inputs)

        if f!=channels:
            t1 = conv(f, 1, stride=1, act=None, pad='valid') (inputs) # identity mapping
        else:
            t1 = inputs

        out = Add()([t1, cs]) # t1 + c2
        out = LeakyReLU(0.1) (out)
        if dropout>0:
            out = Dropout(dropout) (out)
        return out
    return block

def up_bilinear():
    def block(x):
        h, w = K.int_shape(x)[-3:-1] 
        x = Lambda(lambda img: tf.image.resize_bilinear(img, (h*2, w*2), align_corners=True)) (x)
        return x
    return block

def residual_discriminator(h=128, w=128, c=3, k=4, dropout_rate=0.1, as_classifier=0):

    inputs = Input(shape=(h,w,c)) # 32x32@c

    # block 1:
    x = conv(32, k, 1, pad='same') (inputs) # 32x32@32. stride=1 -> reduce checkboard artifacts
    x = LeakyReLU(0.2) (x)
    x = Dropout(dropout_rate) (x)
    x = conv(64, k, 2, pad='same') (x) # 16x16@64
    x = LeakyReLU(0.2) (x)
    x = Dropout(dropout_rate) (x)
    
    # block 2:
    x = conv(128, k, 2, pad='same') (x) # 8x8@128
    x = LeakyReLU(0.2) (x)
    x = Dropout(dropout_rate) (x)
    
    # block 3:
    x = conv(256, k, 2) (x) # 4x4@256
    x = LeakyReLU(0.2) (x)
    x = Dropout(dropout_rate) (x)
    
    # block 3:
    x = conv(256, k, 2) (x) # 2x2@256
    x = LeakyReLU(0.2) (x)
    x = Dropout(dropout_rate) (x)
    
    # block 4:
    x = _res_conv(512, k, dropout_rate) (x) # 2x2@512
    
    hidden = Flatten() (x) # 2*2*512
    
    if as_classifier>0:
        out = Dense(as_classifier, kernel_regularizer=l2(0.001), kernel_initializer='he_normal', activation='softmax') (hidden)
    else:
        out = Dense(1, kernel_regularizer=l2(0.001), kernel_initializer='he_normal') (hidden)
    return Model([inputs], [out])

def residual_decoder(h, w, c=3, k=4, latent_dim=2, dropout_rate=0.1):

    inputs_ = Input(shape=(latent_dim,))
    
    hidden = inputs_
    
    transform = Dense(h*w*512, kernel_regularizer=l2(0.001)) (hidden)
    transform = LeakyReLU(0.1) (transform) # more nonlinearity
    reshape = Reshape((h,w,512)) (transform)

    x = reshape # 2x2@512
    x = Dropout(dropout_rate) (x) # prevent overfitting
    
    x = up_bilinear() (x) # 4x4@512
    x = Conv2DTranspose(128, k, padding='same') (x) # 4x4@128
    x = LeakyReLU(0.2) (x)
    
    x = up_bilinear() (x) # 8x8@128
    x = Conv2DTranspose(128, k, padding='same') (x) # 8x8@128
    x = LeakyReLU(0.2) (x)
    
    x = up_bilinear() (x) # 16x16@128
    x = Conv2DTranspose(64, k, padding='same') (x)  # 16x16@64
    x = LeakyReLU(0.2) (x)
    
    x = _res_conv(64, k, dropout_rate) (x) # 16x16@64
    
    x = PixelShuffler() (x) # 32x32@16
    x = Conv2DTranspose(32, k, padding='same') (x)  # 32x32@32
    x = LeakyReLU(0.2) (x)
    
    x = _res_conv(32, k, dropout_rate) (x) # 32x32@32
    
    outputs = conv(c, k, 1, act='tanh') (x) # 32x32@c

    model = Model([inputs_], [outputs])
    return model

def build_gan(h=128, w=128, c=3, latent_dim=2, epsilon_std=1.0, dropout_rate=0.1, GRADIENT_PENALTY_WEIGHT=10):
    
    optimizer_g = AdamWithWeightnorm(lr=0.0001, beta_1=0.5)
    optimizer_d = AdamWithWeightnorm(lr=0.0001, beta_1=0.5)
    
    t_h, t_w = h//16, w//16
    generator = residual_decoder(t_h, t_w, c=c, latent_dim=latent_dim, dropout_rate=dropout_rate)
    
    discriminator = residual_discriminator(h=h,w=w,c=c,dropout_rate=dropout_rate)
    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False
    
    generator_input = Input(shape=(latent_dim,))
    generator_layers = generator(generator_input)
    
    discriminator_layers_for_generator = discriminator(generator_layers)
    generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
    generator_model.add_loss(K.mean(discriminator_layers_for_generator))
    generator_model.compile(optimizer=optimizer_g, loss=None)

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

    averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
    averaged_samples_out = discriminator(averaged_samples)
    
    discriminator_model = Model([real_samples, generator_input_for_discriminator], [discriminator_output_from_real_samples, discriminator_output_from_generator, averaged_samples_out])
    discriminator_model.add_loss(K.mean(discriminator_output_from_real_samples) - K.mean(discriminator_output_from_generator) + gradient_penalty_loss(averaged_samples_out, averaged_samples, GRADIENT_PENALTY_WEIGHT))
    discriminator_model.compile(optimizer=optimizer_d, loss=None)

    return generator_model, discriminator_model, generator, discriminator

def wgangp_conditional(h=128, w=128, c=3, latent_dim=2, condition_dim=10, epsilon_std=1.0, dropout_rate=0.1, GRADIENT_PENALTY_WEIGHT=10):
    
    optimizer_g = AdamWithWeightnorm(lr=0.0001, beta_1=0.5)
    optimizer_d = AdamWithWeightnorm(lr=0.0001, beta_1=0.5)
    optimizer_c = AdamWithWeightnorm(lr=0.0001, beta_1=0.5)
    
    t_h, t_w = h//16, w//16
    generator = residual_decoder(t_h, t_w, c=c, latent_dim=latent_dim+condition_dim, dropout_rate=dropout_rate)
    
    discriminator = residual_discriminator(h=h,w=w,c=c,dropout_rate=dropout_rate)
    classifier = residual_discriminator(h=h,w=w,c=c,dropout_rate=dropout_rate, as_classifier=condition_dim)
    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False
    for layer in classifier.layers:
        layer.trainable = False
    classifier.trainable = False
    
    generator_input = Input(shape=(latent_dim+condition_dim,))
    generator_layers = generator(generator_input)
    
    discriminator_layers_for_generator = discriminator(generator_layers)
    classifier_layers_for_generator    = classifier(generator_layers)
    
    generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator, classifier_layers_for_generator])
    generator_model.add_loss(K.mean(discriminator_layers_for_generator))
    generator_model.compile(optimizer=optimizer_g, loss=[None, 'categorical_crossentropy'])

    # Now that the generator_model is compiled, we can make the discriminator layers trainable.
    for layer in discriminator.layers:
        layer.trainable = True
    for layer in classifier.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    discriminator.trainable = True
    classifier.trainable = True
    generator.trainable = False

    # The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
    # The noise seed is run through the generator model to get generated images. Both real and generated images
    # are then run through the discriminator. Although we could concatenate the real and generated images into a
    # single tensor, we don't (see model compilation for why).
    real_samples = Input(shape=(h, w, c))
    generator_input_for_discriminator = Input(shape=(latent_dim+condition_dim,))
    generated_samples_for_discriminator = generator(generator_input_for_discriminator)
    discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
    discriminator_output_from_real_samples = discriminator(real_samples)
    
    classifier_output_from_real_samples = classifier(real_samples)

    averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
    averaged_samples_out = discriminator(averaged_samples)
    
    discriminator_model = Model([real_samples, generator_input_for_discriminator], [discriminator_output_from_real_samples, discriminator_output_from_generator, averaged_samples_out])
    discriminator_model.add_loss(K.mean(discriminator_output_from_real_samples) - K.mean(discriminator_output_from_generator) + gradient_penalty_loss(averaged_samples_out, averaged_samples, GRADIENT_PENALTY_WEIGHT))
    discriminator_model.compile(optimizer=optimizer_d, loss=None)
    
    classifier_model = Model([real_samples], [classifier_output_from_real_samples])
    classifier_model.compile(optimizer=optimizer_c, loss='categorical_crossentropy')

    return generator_model, discriminator_model, classifier_model, generator, discriminator, classifier

