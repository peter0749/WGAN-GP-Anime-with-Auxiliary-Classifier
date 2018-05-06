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

def residual_discriminator(h=128, w=128, c=3, k=4, dropout_rate=0.1):

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
    
    out = Dense(1, kernel_regularizer=l2(0.001), kernel_initializer='he_normal') (hidden)
    model = Model([inputs], [out])
    return model

def residual_encoder(h=128, w=128, c=3, k=4, latent_dim=2, epsilon_std=1.0, dropout_rate=0.1):

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

    z_mean =    Dense(latent_dim, kernel_regularizer=l2(0.001))(hidden)
    z_log_var = Dense(latent_dim, kernel_regularizer=l2(0.001))(hidden)

    z = Lambda(sampling, output_shape=(latent_dim,), arguments={'latent_dim':latent_dim, 'epsilon_std':epsilon_std}) ([z_mean, z_log_var])
    model = Model([inputs], [z, z_mean, z_log_var])
    return model, int(x.shape[1]), int(x.shape[2]) # h, w

def residual_decoder(h, w, c=3, k=4, latent_dim=2, dropout_rate=0.1):

    inputs_ = Input(shape=(latent_dim,))
    
    hidden = inputs_
    
    transform = Dense(h*w*512, kernel_regularizer=l2(0.001)) (hidden)
    transform = LeakyReLU(0.1) (transform) # more nonlinearity
    reshape = Reshape((h,w,512)) (transform)

    x = reshape # 2x2@512
    x = Dropout(dropout_rate) (x) # prevent overfitting
    
    x = UpSampling2D((2,2)) (x) # 4x4@512
    # x = PixelShuffler() (x) # 4x4@128
    x = Conv2DTranspose(128, k, padding='same') (x) # 4x4@128
    x = LeakyReLU(0.2) (x)
    
    x = UpSampling2D((2,2)) (x) # 8x8@128
    # x = PixelShuffler() (x) # 8x8@32
    x = Conv2DTranspose(128, k, padding='same') (x) # 8x8@128
    x = LeakyReLU(0.2) (x)
    
    x = UpSampling2D((2,2)) (x) # 16x16@128
    # x = PixelShuffler() (x) # 16x16@32
    x = Conv2DTranspose(64, k, padding='same') (x)  # 16x16@64
    x = LeakyReLU(0.2) (x)
    
    x = _res_conv(64, k, dropout_rate) (x) # 16x16@64
    
    # x = UpSampling2D((2,2)) (x) # 32x32@64
    x = PixelShuffler() (x) # 32x32@16
    x = Conv2DTranspose(32, k, padding='same') (x)  # 32x32@32
    x = LeakyReLU(0.2) (x)
    
    x = _res_conv(32, k, dropout_rate) (x) # 32x32@32
    
    outputs = Conv2DTranspose(c, k, padding='same', activation='tanh') (x) # 32x32@c

    model = Model([inputs_], [outputs])
    return model

def residual_ae(h=128, w=128, c_in=3, c_out=3, k=4, dropout_rate=0.1):
    inputs = Input(shape=(h,w,c_in)) # 32x32@c

    # block 1:
    x = conv(32, k, 1, pad='same') (inputs) # 32x32@32. stride=1 -> reduce checkboard artifacts
    x = LeakyReLU(0.2) (x)
    x = Dropout(dropout_rate) (x)
    x = conv(64, k, 2, pad='same') (x) # 16x16@64
    x = LeakyReLU(0.2) (x)
    x = Dropout(dropout_rate) (x)
    
    # block 2:
    x = _res_conv(128, k, dropout_rate) (x) # 16x16@128
    x = conv(128, k, 2, pad='same') (x) # 8x8@128
    x = LeakyReLU(0.2) (x)
    x = Dropout(dropout_rate) (x)
    
    # block 3:
    x = _res_conv(256, k, dropout_rate) (x) # 8x8@256
    x = conv(256, k, 2) (x) # 4x4@256
    x = LeakyReLU(0.2) (x)
    x = Dropout(dropout_rate) (x)
    
    # block 3:
    x = _res_conv(256, k, dropout_rate) (x) # 4x4@256
    x = conv(256, k, 2) (x) # 2x2@256
    x = LeakyReLU(0.2) (x)
    x = Dropout(dropout_rate) (x)
    
    # block 4:
    x = _res_conv(512, k, dropout_rate) (x) # 2x2@512
    x = _res_conv(512, k, dropout_rate) (x) # 2x2@512
    
    # block 5:
    x = UpSampling2D((2,2)) (x) # 4x4@512
    # x = PixelShuffler() (x) # 4x4@128
    x = Conv2DTranspose(128, k, padding='same') (x) # 4x4@128
    x = LeakyReLU(0.2) (x)
    
    # block 6:
    x = UpSampling2D((2,2)) (x) # 8x8@128
    # x = PixelShuffler() (x) # 8x8@32
    x = Conv2DTranspose(128, k, padding='same') (x) # 8x8@128
    x = LeakyReLU(0.2) (x)
    
    # block 7:
    x = UpSampling2D((2,2)) (x) # 16x16@128
    # x = PixelShuffler() (x) # 16x16@32
    x = Conv2DTranspose(64, k, padding='same') (x)  # 16x16@64
    x = LeakyReLU(0.2) (x)
    
    # block 8:
    x = _res_conv(64, k, dropout_rate) (x) # 16x16@64
    
    # block 9:
    # x = UpSampling2D((2,2)) (x) # 32x32@64
    x = PixelShuffler() (x) # 32x32@16
    x = Conv2DTranspose(32, k, padding='same') (x)  # 32x32@32
    x = LeakyReLU(0.2) (x)
    
    # block 10:
    x = _res_conv(32, k, dropout_rate) (x) # 32x32@32
    
    outputs = Conv2DTranspose(c_out, k, padding='same', activation='tanh') (x) # 32x32@c
    return Model([inputs], [outputs])

def build_residual_vae(h=128, w=128, c=3, latent_dim=2, epsilon_std=1.0, dropout_rate=0.1):

    optimizer = AdamWithWeightnorm(lr=0.0001, beta_1=0.5)
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
    
    optimizer_g = AdamWithWeightnorm(lr=0.0001, beta_1=0.5)
    optimizer_d = AdamWithWeightnorm(lr=0.0001, beta_1=0.5)
    
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
    generator_model.add_loss(K.mean(discriminator_layers_for_generator))
    generator_model.compile(optimizer=optimizer_g, loss=None)
    
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
        vae_model.compile(optimizer=optimizer_g, loss=None)

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

    return (generator_model, discriminator_model, vae_model, encoder_model, generator, discriminator) if use_vae else (generator_model, discriminator_model, generator, discriminator)  

def build_cyclewgan(h=128, w=128, c_A=3, c_B=3, epsilon_std=1.0, dropout_rate=0.1, GRADIENT_PENALTY_WEIGHT=10, batch_size=8, cyclic_loss_w=10):
    
    optimizer_gA = AdamWithWeightnorm(lr=0.0001, beta_1=0.5)
    optimizer_gB = AdamWithWeightnorm(lr=0.0001, beta_1=0.5)
    optimizer_dA = AdamWithWeightnorm(lr=0.0001, beta_1=0.5)
    optimizer_dB = AdamWithWeightnorm(lr=0.0001, beta_1=0.5)
    
    generator_A = residual_ae(h=h, w=w, c_in=c_A, c_out=c_B, dropout_rate=dropout_rate) # A to B
    generator_B = residual_ae(h=h, w=w, c_in=c_B, c_out=c_A, dropout_rate=dropout_rate) # B to A
    
    discriminator_A = residual_discriminator(h=h,w=w,c=c_A,dropout_rate=dropout_rate) # distinguish A
    discriminator_B = residual_discriminator(h=h,w=w,c=c_B,dropout_rate=dropout_rate) # .. B
    for layer in discriminator_A.layers:
        layer.trainable = False
    discriminator_A.trainable = False
    for layer in discriminator_B.layers:
        layer.trainable = False
    discriminator_B.trainable = False
    
    generator_A_input  = Input(shape=(h, w, c_A))
    generator_A_layers = generator_A(generator_A_input) # A->B
    generator_AB_layers= generator_B(generator_A_layers)# B->A
    
    generator_B_input  = Input(shape=(h, w, c_B))
    generator_B_layers = generator_B(generator_B_input) # B->A
    generator_BA_layers= generator_A(generator_B_layers)# A->B
    
    discriminator_B_layers_for_generator_A = discriminator_B(generator_A_layers) # if A->B looks like B
    discriminator_A_layers_for_generator_A = discriminator_A(generator_AB_layers)# if A->B->A looks like A
    
    discriminator_A_layers_for_generator_B = discriminator_A(generator_B_layers) # if B->A looks like A
    discriminator_B_layers_for_generator_B = discriminator_B(generator_BA_layers)# if B->A->B looks like B
    
    generator_A_model = Model(inputs=[generator_A_input], outputs=[discriminator_B_layers_for_generator_A, discriminator_A_layers_for_generator_A])
    generator_A_model.add_loss(K.mean(discriminator_B_layers_for_generator_A), inputs=[generator_A])
    generator_A_model.add_loss(K.mean(discriminator_A_layers_for_generator_A))
    generator_A_model.add_loss(cyclic_loss_w*K.mean(K.abs(generator_A_input - generator_AB_layers)))
    # generator_A_model.add_loss(K.mean(discriminator_B_layers_for_generator_A) + K.mean(discriminator_A_layers_for_generator_A) + cyclic_loss_w*K.mean(K.abs(generator_A_input - generator_AB_layers)))
    generator_A_model.compile(optimizer=optimizer_gA, loss=None)
    
    generator_B_model = Model(inputs=[generator_B_input], outputs=[discriminator_A_layers_for_generator_B, discriminator_B_layers_for_generator_B])
    generator_B_model.add_loss(K.mean(discriminator_A_layers_for_generator_B), inputs=[generator_B])
    generator_B_model.add_loss(K.mean(discriminator_B_layers_for_generator_B))
    generator_B_model.add_loss(cyclic_loss_w*K.mean(K.abs(generator_B_input - generator_BA_layers)))
    #generator_B_model.add_loss(K.mean(discriminator_A_layers_for_generator_B) + K.mean(discriminator_B_layers_for_generator_B) + cyclic_loss_w*K.mean(K.abs(generator_B_input - generator_BA_layers)))
    generator_B_model.compile(optimizer=optimizer_gB, loss=None)

    # Now that the generator_model is compiled, we can make the discriminator layers trainable.
    for layer in discriminator_A.layers:
        layer.trainable = True
    discriminator_A.trainable = True
    for layer in discriminator_B.layers:
        layer.trainable = True
    discriminator_B.trainable = True
    
    for layer in generator_A.layers:
        layer.trainable = False
    generator_A.trainable = False
    for layer in generator_B.layers:
        layer.trainable = False
    generator_B.trainable = False

    # The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
    # The noise seed is run through the generator model to get generated images. Both real and generated images
    # are then run through the discriminator. Although we could concatenate the real and generated images into a
    # single tensor, we don't (see model compilation for why).
    real_samples_A = Input(shape=(h, w, c_A))
    real_samples_B = Input(shape=(h, w, c_B))
    
    generator_A_input_for_discriminator_B = Input(shape=(h, w, c_A)) 
    generator_B_input_for_discriminator_A = Input(shape=(h, w, c_B)) 
    
    generated_samples_B_for_discriminator_B = generator_A(generator_A_input_for_discriminator_B) # A->B_fake
    generated_samples_A_for_discriminator_A = generator_B(generator_B_input_for_discriminator_A) # B->A_fake
    
    discriminator_B_output_from_generator_A  = discriminator_B(generated_samples_B_for_discriminator_B) # discriminate A->B_fake
    discriminator_B_output_from_real_samples_B = discriminator_B(real_samples_B) # discriminate B_real

    averaged_samples_B = RandomWeightedAverage()([real_samples_B, generated_samples_B_for_discriminator_B])
    averaged_samples_B_out = discriminator_B(averaged_samples_B)
    
    discriminator_A_output_from_generator_B  = discriminator_A(generated_samples_A_for_discriminator_A) # discriminate B->A_fake
    discriminator_A_output_from_real_samples_A = discriminator_A(real_samples_A) # discriminate A_real

    averaged_samples_A = RandomWeightedAverage()([real_samples_A, generated_samples_A_for_discriminator_A])
    averaged_samples_A_out = discriminator_A(averaged_samples_A)
    
    discriminator_A_model = Model([real_samples_A, generator_B_input_for_discriminator_A], [discriminator_A_output_from_real_samples_A, discriminator_A_output_from_generator_B, averaged_samples_A_out])
    discriminator_A_model.add_loss(K.mean(discriminator_A_output_from_real_samples_A) - K.mean(discriminator_A_output_from_generator_B) + gradient_penalty_loss(averaged_samples_A_out, averaged_samples_A, GRADIENT_PENALTY_WEIGHT))
    discriminator_A_model.compile(optimizer=optimizer_dA, loss=None)
    
    discriminator_B_model = Model([real_samples_B, generator_A_input_for_discriminator_B], [discriminator_B_output_from_real_samples_B, discriminator_B_output_from_generator_A, averaged_samples_B_out])
    discriminator_B_model.add_loss(K.mean(discriminator_B_output_from_real_samples_B) - K.mean(discriminator_B_output_from_generator_A) + gradient_penalty_loss(averaged_samples_B_out, averaged_samples_B, GRADIENT_PENALTY_WEIGHT))
    discriminator_B_model.compile(optimizer=optimizer_dB, loss=None)

    return generator_A_model, generator_B_model, discriminator_A_model, discriminator_B_model, generator_A, generator_B, discriminator_A, discriminator_B

if __name__ == '__main__':
    vae, encoder, decoder = build_residual_vae(h=96, w=96, c=3, latent_dim=100, dropout_rate=0.2)
    vae.summary()
    encoder.summary()
    decoder.summary()
