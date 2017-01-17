from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

def Basic_Net_v1(im_rows, im_cols, colors):
    '''.
    Input: rowsxcolx1 image
    Training: should take 10-15 epochs and ~15 minutes to train,
              and should be able to achieve a cross-entropy loss of < 1.0

    @im_rows, @im_cols = the number of rows and columns in the input image
    @colors            = the number of color channels in the image (1 or 3)
    '''
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, subsample=(2,2), border_mode='same',
                            input_shape=(im_rows, im_cols, colors)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(32, 3, 3,  subsample=(2,2), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, subsample=(2,2), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, subsample=(2,2), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def Basic_Net_v2(im_rows, im_cols, colors):
    '''
    Input: rowsxcolsx1 image
    Training: This is the same as v1, just with twice the channels
              (and will take twice as long to train)

    @im_rows, @im_cols = the number of rows and columns in the input image
    @colors            = the number of color channels in the image (1 or 3)
    '''
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, subsample=(2,2), border_mode='same',
                            input_shape=(im_rows, im_cols, colors)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3,  subsample=(2,2), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3, subsample=(2,2), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3, subsample=(2,2), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def Basic_Net_v3(im_rows, im_cols, colors):
    '''
    Input: rowsxcolsx1 image
    Training: This is the same as v1, just with twice the channels
              (and will take twice as long to train)

    @im_rows, @im_cols = the number of rows and columns in the input image
    @colors            = the number of color channels in the image (1 or 3)
    '''
    model1 = Sequential()
    model1.add(Convolution2D(32, 3, 3, subsample=(2,2), border_mode='same',
                            input_shape=(im_rows, im_cols, colors)))

    model2 = Sequential()
    model2.add(Convolution2D(32, 3, 3, subsample=(2,2), border_mode='same',
                            input_shape=(im_rows / 2, im_cols / 2, colors)))

    model = Sequential()
    model.add(Merge([model1, model2], mode='concat', concat_axis=2))

    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3,  subsample=(2,2), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3, subsample=(2,2), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3, subsample=(2,2), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3, subsample=(2,2), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model
