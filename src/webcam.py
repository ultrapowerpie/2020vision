import os, glob, pickle
import math, cv2, sys, vlc, time
import numpy as np

from keras.utils import np_utils
from keras.models import model_from_json, Model
from keras.optimizers import SGD
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier

import basic_net, fancy_net
import util
import tensorflow as tf
import keras

im_cols, im_rows = 64, 64

binary = True

if binary:
    cats = 2
else:
    cats = 10

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

p = vlc.MediaPlayer('static_data/beep-02.mp3')

model = util.read_model(sys.argv[1])

# K-Nearest Neighbors
# interm_layer_model = util.build_interm_model(model)
# interm_train = interm_layer_model.predict(x_train, verbose=1)
#
# knn = KNeighborsClassifier(n_neighbors=n_neighbors)
# knn.fit(interm_train, y_train)

last5 = 0
start = time.time()

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()

    resized = cv2.resize(frame, (im_cols, im_rows))

    cv2.imshow("preview2", resized)

    x = np.resize(resized, (1, im_cols, im_rows, 3))

    #  interm_valid = interm_layer_model.predict(x)
    #  knn_prediction = knn.predict(interm_valid)
    #  category = np.argmax(knn_prediction)
    softmax = model.predict(x)
    category = np.argmax(softmax)

    if (category == 0) and (last5 < 5):
        last5 += 1
    elif last5 > 0:
        last5 -= 1

    if last5 < 3:
        end = time.time()
        if (end - start) > 1:
            p.stop()
            start = time.time()
            p.play()


    print(category)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
