from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

from sklearn.metrics import log_loss

import util

def create_model_v1(img_rows, img_cols, colors=1):
    nb_classes = 10
    nb_filters = 8
    nb_pool = 2
    nb_conv = 2
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(img_rows, img_cols, colors)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=0, momentum=0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

#AlexNet with batch normalization in Keras
#input image is 224x224
def create_model_v2(img_rows, img_cols, colors=1):

    eta = 0.1
    decay = 0
    momentum = 0
    nesterov = False

    model = Sequential()
    model.add(Convolution2D(64, 11, 11, border_mode='same', input_shape=(img_rows, img_cols, colors)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(128, 7, 7, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(192, 3, 3, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(256, 3, 3, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    sgd = SGD(lr=eta, decay=decay, momentum=momentum, nesterov=nesterov)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def main():
    # input image dimensions
    img_rows, img_cols = 224, 224
    batch_size         = 32
    nb_epoch           = 2
    random_state       = 51
    colors             = 1

    train_data, train_target, driver_id, unique_drivers = util.load_train_data(img_rows, img_cols, colors)
    test_data, test_id = util.load_test_data(img_rows, img_cols, colors)

    yfull_train = dict()
    yfull_test = []
    unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                     'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                     'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                     'p075']
    x_train, y_train, train_index = util.copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
    unique_list_valid = ['p081']
    x_valid, y_valid, test_index = util.copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

    print ('Starting Single Run')
    print ('Split train: ', len(x_train), len(y_train))
    print ('Split valid: ', len(x_valid), len(y_valid))
    print ('Train drivers: ', unique_list_train)
    print ('Test drivers: ', unique_list_valid)

    model = create_model_v2(img_rows, img_cols, colors)
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(x_valid, y_valid))

    score = model.evaluate(x_valid, y_valid, verbose=0)
    print('Score log_loss: ', score[0])

    predictions_valid = model.predict(x_valid, batch_size=128, verbose=1)
    score = log_loss(y_valid, predictions_valid)
    print('Score log_loss: ', score)

    # Store valid predictions
    for i in range(len(test_index)):
        yfull_train[test_index[i]] = predictions_valid[i]

    # Store test predictions
    test_prediction = model.predict(test_data, batch_size=128, verbose=1)
    yfull_test.append(test_prediction)

    print 'Final log_loss: {}, rows: {} cols: {} epoch: {}'.format(score, img_rows, img_cols, nb_epoch)
    # info_string = 'loss_' + str(score) \
    #                 + '_r_' + str(img_rows) \
    #                 + '_c_' + str(img_cols) \
    #                 + '_ep_' + str(nb_epoch)
    #
    # test_res = merge_several_folds_mean(yfull_test, 1)
    # create_submission(test_res, test_id, info_string)

if __name__ == "__main__":
    main()
