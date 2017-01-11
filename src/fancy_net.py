import numpy as np
import math
import keras

from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Activation, Flatten, Dropout, merge
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import SGD

drop = 0.5

def _add_conv(x, channels, nb_filter, subsample=(1, 1)):
    return Conv2D(channels, nb_filter[0], nb_filter[1], subsample=subsample,
                    border_mode='same', init="he_normal", bias=False)(x)


def _add_conv_bn_act(x, channels, subsample=(1, 1)):
    x = _add_conv(x, channels, (3, 3), subsample=subsample)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(drop)(x)

    return x


def _add_conv_bn(x, channels, nb_filter, subsample=(1, 1)):
    x = _add_conv(x, channels, nb_filter, subsample=subsample)
    return BatchNormalization()(x)


def _add_bn_act(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(drop)(x)

    return x


def Fancy_Net_v1(im_rows, im_cols, colors):
    '''
    This is a small convolution-only ResNet, with normal Residuals, as proposed
    in the original ResNet paper, with Dropout added to every layer

    Input: 64x64x1 image
    Training: should take 10-15 epochs and ~50 minutes to train,
              and should be able to achieve a cross-entropy loss of < 0.8

    @im_rows, @im_cols = the number of rows and columns in the input image
    @colors            = the number of color channels in the image (1 or 3)
    '''
    im_input = Input(shape=(im_rows, im_cols, colors))

    channels = int(im_rows/4)
    iterations = 2

    residual = _add_conv_bn(im_input, channels, (1, 1), subsample=(2, 2))
    x = _add_conv_bn_act(im_input, channels, subsample=(2, 2))
    x = _add_conv_bn(x, channels, (3, 3))
    x = merge([x, residual], mode='sum')

    channels *= 2

    for i in range(iterations):

        residual = _add_conv_bn(x, channels, (1, 1), subsample=(2, 2))

        x = Activation('relu')(x)
        x = Dropout(drop)(x)

        x = _add_conv_bn_act(x, channels, subsample=(2, 2))
        x = _add_conv_bn(x, channels, (3, 3))

        x = merge([x, residual], mode='sum')

        # channels *= 2

    x = Activation('relu')(x)
    x = Dropout(drop)(x)
    x = _add_conv_bn_act(x, channels, subsample=(2, 2))

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(10, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(im_input, x)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def Fancy_Net_v2(im_rows, im_cols, colors):
    '''
    This is a small convolution-only ResNet with identity residuals and Dropout
    added to every layer.

    Input: 64x64x1 image
    Training: should take 10-15 epochs and ~50 minutes to train,
              and should be able to achieve a cross-entropy loss of < 0.8

    @im_rows, @im_cols = the number of rows and columns in the input image
    @colors            = the number of color channels in the image (1 or 3)
    '''
    im_input = Input(shape=(im_rows, im_cols, colors))

    channels = int(im_rows/4)
    iterations = 2

    x = _add_conv(im_input, channels, (3, 3), subsample=(2, 2))

    residual = _add_conv(x, 2*channels, (1, 1), subsample=(2, 2))

    x = _add_bn_act(x)
    x = _add_conv_bn_act(x, channels)
    x = _add_conv(x, 2*channels, (3, 3), subsample=(2, 2))

    x = merge([x, residual], mode='sum')

    channels *= 2

    for i in range(iterations):

        residual = _add_conv(x, channels, (1, 1), subsample=(2, 2))

        x = _add_bn_act(x)
        x = _add_conv_bn_act(x, channels)
        x = _add_conv(x, channels, (3, 3), subsample=(2, 2))

        x = merge([x, residual], mode='sum')


    channels *= 2

    x = _add_bn_act(x)
    x = _add_conv_bn_act(x, channels, subsample=(2, 2))

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(2, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(im_input, x)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def Fancy_Net_v3(im_rows, im_cols, colors):
    '''
    This is the entire Xception network, designed for ImageNet, but if you have
    an Nvidia GTX 1080 and 12gb of GDDR5, feel free to go ahead and crush it.
    '''

    im_input = Input(shape=(im_rows, im_cols, colors))

    x = Conv2D(32, 3, 3, bias=False, name='block1_conv1')(im_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, 3, 3, bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(64, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, 3, 3, border_mode='same', bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(64, 3, 3, border_mode='same', bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block2_pool')(x)
    x = merge([x, residual], mode='sum')

    residual = Conv2D(128, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(128, 3, 3, border_mode='same', bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(128, 3, 3, border_mode='same', bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block3_pool')(x)
    x = merge([x, residual], mode='sum')

    residual = Conv2D(192, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(192, 3, 3, border_mode='same', bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(192, 3, 3, border_mode='same', bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block4_pool')(x)
    x = merge([x, residual], mode='sum')

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(192, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(192, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(192, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = merge([x, residual], mode='sum')

    residual = Conv2D(256, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(192, 3, 3, border_mode='same', bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(256, 3, 3, border_mode='same', bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block13_pool')(x)
    x = merge([x, residual], mode='sum')

    x = SeparableConv2D(384, 3, 3, border_mode='same', bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(512, 3, 3, border_mode='same', bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(10, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(im_input, x)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model
