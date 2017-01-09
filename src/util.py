import os, glob, pickle
import math, random, cv2
import numpy as np
np.random.seed(2020)

from keras.utils import np_utils
from keras.models import model_from_json, Model
from keras.optimizers import SGD
from tqdm import tqdm

# limit the number of testing images because we can't handle that much data

train_limit = 3000
test_limit = 4999


# colors = 1 for grayscale, 3 for rgb
def get_im_cv2(path, im_rows, im_cols, colors=1):
    if colors == 1:
        img = cv2.imread(path, 0)
    elif colors == 3:
        img = cv2.imread(path)

    resized = cv2.resize(img, (im_cols, im_rows))
    return resized


def get_driver_data():
    driver_dict = dict()
    path = os.path.join('static_data', 'input', 'driver_imgs_list.csv')
    print('Reading driver data')
    with open(path, 'r') as f:
        for line in f:
            array = line.strip().split(',')
            driver_dict[array[2]] = array[0]

    return driver_dict


def load_train(im_rows, im_cols, colors=1):
    print 'Reading training images...'

    x_train = []
    y_train = []
    driver_id = []
    driver_data = get_driver_data()

    for j in range(10):
        print 'Loading folder c{}...'.format(j)
        path = os.path.join('static_data', 'input', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)

        for i, f in enumerate(tqdm(files)):
            if i > train_limit:
                break
            base = os.path.basename(f)
            img = get_im_cv2(f, im_rows, im_cols, colors)
            x_train.append(img)
            y_train.append(j)
            driver_id.append(driver_data[base])

    unique_drivers = sorted(list(set(driver_id)))
    print 'Unique drivers: {}'.format(len(unique_drivers))
    print unique_drivers
    return x_train, y_train, driver_id, unique_drivers


def load_test(im_rows, im_cols, colors=1):
    print 'Reading testing images...'

    path = os.path.join('static_data', 'input', 'test', '*.jpg')
    files = glob.glob(path)
    random.shuffle(files)

    x_test = []
    x_test_id = []

    for i, f in enumerate(tqdm(files)):
        if i > test_limit:
            break
        base = os.path.basename(f)
        img = get_im_cv2(f, im_rows, im_cols, colors)
        x_test.append(img)
        x_test_id.append(base)

    return x_test, x_test_id

def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    else:
        print "Directory doesn't exist"


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        print 'Cache not found'

    return data

def normalize_data(data, im_rows, im_cols, colors):
    data = np.array(data, dtype=np.uint8)
    data = data.reshape(data.shape[0], im_rows, im_cols, colors)
    data = data.astype('float32')
    data /= 255

    return data

def load_train_data(im_rows, im_cols, colors=1, use_cache=True):
    cache_path = os.path.join('cache', 'train_r_' + str(im_rows) + '_c_' + str(im_cols) + '_t_' + str(colors) + '.dat')
    if os.path.isfile(cache_path) and use_cache:
        print('Restoring training data from cache...')
        (train_data, train_target, driver_id, unique_drivers) = restore_data(cache_path)
    else:
        train_data, train_target, driver_id, unique_drivers = load_train(im_rows, im_cols, colors)
        cache_data((train_data, train_target, driver_id, unique_drivers), cache_path)

    train_data = normalize_data(train_data, im_rows, im_cols, colors)

    train_target = np.array(train_target, dtype=np.uint8)
    train_target = np_utils.to_categorical(train_target, 10)

    print 'Train shape: {}'.format(train_data.shape)

    return train_data, train_target, driver_id, unique_drivers


def load_test_data(im_rows, im_cols, colors=1, use_cache=True):
    cache_path = os.path.join('cache', 'test_r_' + str(im_rows) + '_c_' + str(im_cols) + '_t_' + str(colors) + '.dat')
    if os.path.isfile(cache_path) and use_cache:
        print('Restoring testing from cache...')
        (test_data, test_id) = restore_data(cache_path)
    else:
        test_data, test_id = load_test(im_rows, im_cols, colors)
        cache_data((test_data, test_id), cache_path)

    test_data = normalize_data(test_data, im_rows, im_cols, colors)

    print ('Test shape:', test_data.shape)

    return test_data, test_id


def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    driver_set = set(driver_list)
    for i in range(len(driver_id)):
        if driver_id[i] in driver_set:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)

    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    index = np.array(index, dtype=np.uint32)
    return data, target, index


def save_model(model, name):
    json_string = model.to_json()
    open(os.path.join('cache', name+'_architecture.json'), 'w').write(json_string)
    model.save_weights(os.path.join('cache', name+'_model_weights.h5'), overwrite=True)


def read_model():
    model = model_from_json(open(os.path.join('cache', 'architecture.json')).read())
    model.load_weights(os.path.join('cache', 'model_weights.h5'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def build_interm_model(model):
    model = Model(input=model.input, output=model.layers[-1].output)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model
