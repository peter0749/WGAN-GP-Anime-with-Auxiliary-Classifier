import os
import numpy as np
import keras
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate
from keras.layers import Activation, LeakyReLU, ELU
from keras.regularizers import l2
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from weightnorm import AdamWithWeightnorm as Adam
from keras import backend as K
from models import conv, _res_conv, up_bilinear, set_trainable

def sample_normal(args):
    z_avg, z_log_var = args
    batch_size = K.shape(z_avg)[0]
    z_dims = K.shape(z_avg)[1]
    eps = K.random_normal(shape=(batch_size, z_dims), mean=0.0, stddev=1.0)
    return z_avg + K.exp(z_log_var / 2.0) * eps

class ClassifierLossLayer(Layer):
    __name__ = 'classifier_loss_layer'

    def __init__(self, label_smooth=.9, **kwargs):
        self.is_placeholder = True
        self.label_smooth = label_smooth
        super(ClassifierLossLayer, self).__init__(**kwargs)

    def lossfun(self, c_true, c_pred):
        return K.mean(keras.metrics.categorical_crossentropy(c_true * self.label_smooth, c_pred))

    def call(self, inputs):
        c_true = inputs[0]
        c_pred = inputs[1]
        loss = self.lossfun(c_true, c_pred)
        self.add_loss(K.clip(loss, -100, 100), inputs=inputs)
        return c_true

class DiscriminatorLossLayer(Layer):
    __name__ = 'discriminator_loss_layer'

    def __init__(self, label_smooth=.9, **kwargs):
        self.is_placeholder = True
        self.label_smooth = label_smooth
        super(DiscriminatorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_real, y_fake_f, y_fake_p):
        y_pos = K.ones_like(y_real) * self.label_smooth
        y_neg = K.zeros_like(y_real)
        loss_real = keras.metrics.binary_crossentropy(y_pos, K.clip(y_real, K.epsilon(), 1.-K.epsilon()))
        loss_fake_f = keras.metrics.binary_crossentropy(y_neg, K.clip(y_fake_f, K.epsilon(), 1.-K.epsilon()))
        loss_fake_p = keras.metrics.binary_crossentropy(y_neg, K.clip(y_fake_p, K.epsilon(), 1.-K.epsilon()))
        return K.mean(loss_real + loss_fake_f + loss_fake_p)

    def call(self, inputs):
        y_real = inputs[0]
        y_fake_f = inputs[1]
        y_fake_p = inputs[2]
        loss = self.lossfun(y_real, y_fake_f, y_fake_p)
        self.add_loss(K.clip(loss, -100, 100), inputs=inputs)

        return y_real

class GeneratorLossLayer(Layer):
    __name__ = 'generator_loss_layer'

    def __init__(self, reconstruct_loss='l1', lambda_2=1.0, **kwargs):
        self.is_placeholder = True
        self.reconstruct_loss = reconstruct_loss
        self.lambda_2 = lambda_2
        super(GeneratorLossLayer, self).__init__(**kwargs)

    def lossfun(self, x_r, x_f, f_D_x_f, f_D_x_r, f_C_x_r, f_C_x_f):
        if self.reconstruct_loss == 'l1':
            loss_x = K.mean(K.abs(x_r - x_f))
            loss_d = K.mean(K.abs(f_D_x_r - f_D_x_f))
            loss_c = K.mean(K.abs(f_C_x_r - f_C_x_f))
        elif self.reconstruct_loss == 'l2':
            loss_x = K.mean(K.square(x_r - x_f))
            loss_d = K.mean(K.square(f_D_x_r - f_D_x_f))
            loss_c = K.mean(K.square(f_C_x_r - f_C_x_f))
        else:
            loss_x = K.mean(K.binary_crossentropy(K.clip(x_r*.5+.5, K.epsilon(), 1.-K.epsilon()) , K.clip(x_f*.5+.5, K.epsilon(), 1.-K.epsilon())))
            loss_d = K.mean(K.square(f_D_x_r - f_D_x_f))
            loss_c = K.mean(K.square(f_C_x_r - f_C_x_f))

        return (loss_x + loss_d + loss_c) * self.lambda_2

    def call(self, inputs):
        x_r = inputs[0]
        x_f = inputs[1]
        f_D_x_r = inputs[2]
        f_D_x_f = inputs[3]
        f_C_x_r = inputs[4]
        f_C_x_f = inputs[5]
        loss = self.lossfun(x_r, x_f, f_D_x_r, f_D_x_f, f_C_x_r, f_C_x_f)
        self.add_loss(K.clip(loss, -100, 100), inputs=inputs)

        return x_r

class FeatureMatchingLayer_GD(Layer):
    __name__ = 'feature_matching_layer_GD'

    def __init__(self, lambda_3=1e-3, **kwargs):
        self.is_placeholder = True
        self.lambda_3 = lambda_3
        super(FeatureMatchingLayer_GD, self).__init__(**kwargs)

    def lossfun(self, f1, f2):
        f1_avg = K.mean(f1, axis=0)
        f2_avg = K.mean(f2, axis=0)
        return 0.5 * K.mean(K.square(f1_avg - f2_avg)) * self.lambda_3

    def call(self, inputs):
        f1 = inputs[0]
        f2 = inputs[1]
        loss = self.lossfun(f1, f2)
        self.add_loss(K.clip(loss, -100, 100), inputs=inputs)
        return f1

class FeatureMatchingLayer_GC(Layer):
    __name__ = 'feature_matching_layer_GC'

    def __init__(self, lambda_4=1e-3, **kwargs):
        self.is_placeholder = True
        self.lambda_4 = lambda_4
        super(FeatureMatchingLayer_GC, self).__init__(**kwargs)
    def call(self, inputs):
        f1 = K.batch_flatten(inputs[0]) # (batch_size, ?)
        f2 = K.batch_flatten(inputs[1]) # (batch_size, ?)
        y   = inputs[2] # (batch_size, class_n)
        y_p = inputs[3] # (batch_size, class_n)
        
        f1_ma = K.mean(K.dot(K.transpose(f1), y  ), axis=0)
        f2_ma = K.mean(K.dot(K.transpose(f2), y_p), axis=0)
        
        loss = 0.5 * K.mean(K.square(f1_ma-f2_ma)) * self.lambda_4
        self.add_loss(K.clip(loss, -100, 100), inputs=inputs)
        return f1

class KLLossLayer(Layer):
    __name__ = 'kl_loss_layer'

    def __init__(self, lambda_1=1.0, **kwargs):
        self.is_placeholder = True
        self.lambda_1 = lambda_1
        super(KLLossLayer, self).__init__(**kwargs)

    def lossfun(self, z_avg, z_log_var):
        kl_loss = K.clip(-0.5 * K.mean(1.0 + z_log_var - K.square(z_avg) - K.exp(z_log_var)), -100.0, 100.0)
        return kl_loss * self.lambda_1

    def call(self, inputs):
        z_avg = inputs[0]
        z_log_var = inputs[1]
        loss = self.lossfun(z_avg, z_log_var)
        self.add_loss(loss, inputs=inputs)

        return z_avg

class CVAEGAN(object):
    def __init__(self,
        input_shape=(64, 64, 3),
        num_attrs=40,
        z_dims = 128,     # Default setting in paper:
        lambda_1 = 1.0,   # 3.0
        lambda_2 = 1.0,   # 1.0
        lambda_3 = 1.0,   # 1e-3
        lambda_4 = 1.5,   # 1e-3
        reconstruct_loss = 'l1',
        name='cvaegan',
        **kwargs
    ):

        self.input_shape = input_shape
        self.num_attrs = num_attrs
        self.z_dims = z_dims
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.reconstruct_loss = reconstruct_loss

        self.f_enc = None
        self.f_dec = None
        self.f_dis = None
        self.f_cls = None
        self.enc_trainer = None
        self.dec_trainer = None
        self.dis_trainer = None
        self.cls_trainer = None

        self.build_model()

    def train_on_batch(self, x_batch):
        x_r, c = x_batch

        batchsize = len(x_r)
        z_p = np.random.normal(size=(batchsize, self.z_dims)).astype('float32')
        c_p = keras.utils.to_categorical(np.random.randint(self.num_attrs, size=batchsize), self.num_attrs)

        # Train classifier
        c_loss = self.cls_trainer.train_on_batch([x_r, c], None)

        # Train discriminator
        d_loss = self.dis_trainer.train_on_batch([x_r, c, c_p, z_p], None)

        # Train generator
        g_loss = self.dec_trainer.train_on_batch([x_r, c, c_p, z_p], None)
        
        # Train autoencoder
        e_loss = self.enc_trainer.train_on_batch([x_r, c, z_p], None)
        
        loss = {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'c_loss': c_loss,
            'e_loss': e_loss
        }
        return loss

    def predict(self, z_samples):
        return self.f_dec.predict(z_samples)

    def build_model(self):
        self.f_enc = self.build_encoder(output_dims=self.z_dims*2)
        self.f_dec = self.build_decoder()
        self.f_dis = self.build_discriminator()
        self.f_cls = self.build_classifier()

        # Algorithm
        x_r = Input(shape=self.input_shape)
        c = Input(shape=(self.num_attrs,))
        c_p = Input(shape=(self.num_attrs,))
        z_params = self.f_enc([x_r, c])

        z_avg = Lambda(lambda x: x[:, :self.z_dims], output_shape=(self.z_dims,))(z_params)
        z_log_var = Lambda(lambda x: x[:, self.z_dims:], output_shape=(self.z_dims,))(z_params)
        z = Lambda(sample_normal, output_shape=(self.z_dims,))([z_avg, z_log_var])

        kl_loss = KLLossLayer(lambda_1 = self.lambda_1)([z_avg, z_log_var])

        z_p = Input(shape=(self.z_dims,))

        x_f = self.f_dec([z, c])
        x_p = self.f_dec([z_p, c_p])

        y_r, f_D_x_r = self.f_dis(x_r)
        y_f, f_D_x_f = self.f_dis(x_f)
        y_p, f_D_x_p = self.f_dis(x_p)

        d_loss = DiscriminatorLossLayer()([y_r, y_f, y_p])

        c_r, f_C_x_r = self.f_cls(x_r)
        c_f, f_C_x_f = self.f_cls(x_f)
        c_p_ , f_C_x_p = self.f_cls(x_p)

        g_loss = GeneratorLossLayer(reconstruct_loss=self.reconstruct_loss, lambda_2 = self.lambda_2)([x_r, x_f, f_D_x_r, f_D_x_f, f_C_x_r, f_C_x_f])
        gd_loss = FeatureMatchingLayer_GD(lambda_3 = self.lambda_3)([f_D_x_r, f_D_x_p])
        gc_loss = FeatureMatchingLayer_GC(lambda_4 = self.lambda_4)([f_C_x_r, f_C_x_p, c, c_p])

        c_loss = ClassifierLossLayer()([c, c_r])

        # Build classifier trainer
        set_trainable(self.f_enc, False)
        set_trainable(self.f_dec, False)
        set_trainable(self.f_dis, False)
        set_trainable(self.f_cls, True)

        self.cls_trainer = Model(inputs=[x_r, c],
                                 outputs=[c_loss])
        self.cls_trainer.compile(loss=None,
                                 optimizer=Adam(lr=5e-5, beta_1=0.5, clipvalue=0.8))
        self.cls_trainer.summary()

        # Build discriminator trainer
        set_trainable(self.f_enc, False)
        set_trainable(self.f_dec, False)
        set_trainable(self.f_dis, True)
        set_trainable(self.f_cls, False)

        self.dis_trainer = Model(inputs=[x_r, c, c_p, z_p],
                                 outputs=[d_loss])
        self.dis_trainer.compile(loss=None,
                                 optimizer=Adam(lr=5e-5, beta_1=0.5, clipvalue=0.8))
        self.dis_trainer.summary()

        # Build generator trainer
        set_trainable(self.f_enc, False)
        set_trainable(self.f_dec, True)
        set_trainable(self.f_dis, False)
        set_trainable(self.f_cls, False)

        self.dec_trainer = Model(inputs=[x_r, c, c_p, z_p],
                                 outputs=[g_loss, gd_loss, gc_loss])
        self.dec_trainer.compile(loss=None,
                                 optimizer=Adam(lr=5e-5, beta_1=0.5, clipvalue=0.8))

        # Build autoencoder
        set_trainable(self.f_enc, True)
        set_trainable(self.f_dec, False)
        set_trainable(self.f_dis, False)
        set_trainable(self.f_cls, False)

        self.enc_trainer = Model(inputs=[x_r, c, z_p],
                                outputs=[g_loss, kl_loss])
        self.enc_trainer.compile(loss=None,
                                 optimizer=Adam(lr=2e-5, beta_1=0.5, clipvalue=0.8))
        self.enc_trainer.summary()

    def build_encoder(self, output_dims, k=4):
        x_inputs = Input(shape=self.input_shape)
        c_inputs = Input(shape=(self.num_attrs,))

        c = Reshape((1, 1, self.num_attrs))(c_inputs)
        c = UpSampling2D(size=self.input_shape[:2])(c)
        x = Concatenate(axis=-1)([x_inputs, c])

        x = conv(f=128, k=k, stride=2)(x)
        x = LeakyReLU(0.2) (x)
        x = conv(f=256, k=k, stride=2)(x)
        x = LeakyReLU(0.2) (x)
        x = conv(f=256, k=k, stride=2)(x)
        x = LeakyReLU(0.2) (x)
        x = conv(f=512, k=k, stride=2)(x)
        x = LeakyReLU(0.2) (x)

        x = Flatten()(x)
        x = Dense(1024, kernel_regularizer=l2(0.001))(x)
        x = LeakyReLU(0.2) (x)

        x = Dense(output_dims, kernel_regularizer=l2(0.001))(x)

        return Model([x_inputs, c_inputs], x)

    def build_decoder(self, k=4):
        z_inputs = Input(shape=(self.z_dims,))
        c_inputs = Input(shape=(self.num_attrs,))
        z = Concatenate()([z_inputs, c_inputs])

        w = self.input_shape[1] // (2 ** 4)
        h = self.input_shape[0] // (2 ** 4)
        x = Dense(h * w * 512, kernel_regularizer=l2(0.001))(z)
        # x = BatchNormalization()(x)
        x = LeakyReLU(0.2) (x)

        x = Reshape((h, w, 512))(x)

        x = up_bilinear() (x) 
        x = Conv2DTranspose(512, k, padding='same') (x) 
        x = LeakyReLU(0.2) (x)
        x = up_bilinear() (x) 
        x = Conv2DTranspose(256, k, padding='same') (x) 
        x = LeakyReLU(0.2) (x)
        x = up_bilinear() (x) 
        x = Conv2DTranspose(256, k, padding='same') (x) 
        x = LeakyReLU(0.2) (x)
        x = up_bilinear() (x) 
        x = Conv2DTranspose(128, k, padding='same') (x) 
        x = LeakyReLU(0.2) (x)

        d = self.input_shape[2]
        
        x = Conv2DTranspose(d, k, padding='same', activation='tanh') (x) 
        return Model([z_inputs, c_inputs], x)

    def build_discriminator(self, k=4):
        inputs = Input(shape=self.input_shape)
        
        x = inputs

        x = conv(f=128, k=k, stride=2)(x)
        x = LeakyReLU(0.2) (x)
        x = conv(f=256, k=k, stride=2)(x)
        x = LeakyReLU(0.2) (x)
        x = conv(f=256, k=k, stride=2)(x)
        x = LeakyReLU(0.2) (x)
        x = conv(f=512, k=k, stride=2)(x)
        x = LeakyReLU(0.2) (x)

        f = Flatten()(x)
        x = Dense(1024, kernel_regularizer=l2(0.001))(f)
        x = LeakyReLU(0.2) (x)

        x = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001))(x)

        return Model(inputs, [x, f])

    def build_classifier(self, k=4):
        inputs = Input(shape=self.input_shape)

        x = inputs
        
        x = conv(f=128, k=k, stride=2)(x)
        x = LeakyReLU(0.2) (x)
        x = conv(f=256, k=k, stride=2)(x)
        x = LeakyReLU(0.2) (x)
        x = conv(f=256, k=k, stride=2)(x)
        x = LeakyReLU(0.2) (x)
        x = conv(f=512, k=k, stride=2)(x)
        x = LeakyReLU(0.2) (x)

        f = Flatten()(x)
        x = Dense(1024, kernel_regularizer=l2(0.001))(f)
        x = LeakyReLU(0.2) (x)

        x = Dense(self.num_attrs, activation='softmax', kernel_regularizer=l2(0.001))(x)

        return Model(inputs, [x, f])
    
    def save_models(self, prefix, idx):
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        self.f_enc.save(os.path.join(prefix, 'iteration_{:d}_encoder.h5'.format(idx)))
        self.f_dec.save(os.path.join(prefix, 'iteration_{:d}_decoder.h5'.format(idx)))
        self.f_dis.save(os.path.join(prefix, 'iteration_{:d}_discriminator.h5'.format(idx)))
        self.f_cls.save(os.path.join(prefix, 'iteration_{:d}_classifier.h5'.format(idx)))
    
    def return_models(self):
        return self.f_enc, self.f_dec, self.f_dis, self.f_cls