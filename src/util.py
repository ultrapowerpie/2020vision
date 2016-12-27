import numpy as np
np.random.seed(2020)

import os
import glob
import cv2
import math
import pickle

from tqdm import tqdm

# color_type = "grayscale", "rgb"
def get_im_cv2(path, img_rows, img_cols, color_type="grayscale"):
    if color_type == "grayscale":
        img = cv2.imread(path, 0)
    elif color_type == "rgb":
        img = cv2.imread(path)

    resized = cv2.resize(img, (img_cols, img_rows))
    return resized


def get_driver_data():
    driver_dict = dict()
    path = os.path.join('static_data', 'input', 'driver_imgs_list.csv')
    print('Read drivers data')
    with open(path, 'r') as f:
        for line in f:
            array = line.strip().split(',')
            driver_dict[array[2]] = array[0]

    return driver_dict


def load_train(img_rows, img_cols, color_type="grayscale"):
    x_train = []
    y_train = []
    driver_id = []

    driver_data = get_driver_data()

    print('Read train images')
    for j in range(10):
        print 'Load folder c{}'.format(j)
        path = os.path.join('static_data', 'input', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for f in tqdm(files):
            base = os.path.basename(f)
            img = get_im_cv2(f, img_rows, img_cols, color_type)
            x_train.append(img)
            y_train.append(j)
            driver_id.append(driver_data[base])

    unique_drivers = sorted(list(set(driver_id)))
    print 'Unique drivers: {}'.format(len(unique_drivers))
    print unique_drivers
    return x_train, y_train, driver_id, unique_drivers


def load_test(img_rows, img_cols, color_type="grayscale"):
    print('Read test images')
    path = os.path.join('static_data', 'input', 'test', '*.jpg')
    files = glob.glob(path)
    x_test = []
    x_test_id = []
    for f in tqdm(files):
        base = os.path.basename(f)
        img = get_im_cv2(f, img_rows, img_cols, color_type)
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

def normalize_data(data, color_type, img_rows, img_cols):
    data = np.array(data, dtype=np.uint8)
    data = train_data.reshape(data.shape[0], color_type, img_rows, img_cols)
    data = data.astype('float32')
    data /= 255

def read_and_normalize_train_data(img_rows, img_cols, color_type="grayscale", use_cache=True):
    cache_path = os.path.join('cache', 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + color_type + '.dat')
    if os.path.isfile(cache_path) and use_cache:
        print('Restoring training data from cache...')
        (train_data, train_target, driver_id, unique_drivers) = restore_data(cache_path)
    else:
        train_data, train_target, driver_id, unique_drivers = load_train(img_rows, img_cols, color_type)
        cache_data((train_data, train_target, driver_id, unique_drivers), cache_path)

    train_data = normalize_data(train_data, color_type, img_rows, img_cols)

    train_target = np.array(train_target, dtype=np.uint8)
    train_target = np_utils.to_categorical(train_target, 10)

    print 'Train shape:'
    print train_data.shape

    return train_data, train_target, driver_id, unique_drivers


def read_and_normalize_test_data(img_rows, img_cols, color_type="grayscale", use_cache=True):
    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + color_type + '.dat')
    if os.path.isfile(cache_path) and use_cache:
        print('Restoring testing from cache...')
        (test_data, test_id) = restore_data(cache_path)
    else:
        test_data, test_id = load_test(img_rows, img_cols, color_type)
        cache_data((test_data, test_id), cache_path)

    test_data = normalize_data(test_data, color_type, img_rows, img_cols)

    print 'Test shape:'
    print test_data.shape

    return test_data, test_id

load_train(299, 299)
