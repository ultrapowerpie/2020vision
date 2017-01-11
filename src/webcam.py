import os, glob, pickle
import math, cv2, sys, vlc, time
import numpy as np

from keras.utils import np_utils
from keras.models import model_from_json, Model
from keras.optimizers import SGD

import basic_net, fancy_net
import util
import tensorflow as tf
import keras

nb_models = 1
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

models = []
for i in range(nb_models):
    if binary:
        models.append(util.read_model(sys.argv[1]+'_'+str(i)+'_b'))
    else:
        models.append(util.read_model(sys.argv[1]+'_'+str(i)))

last5 = 0
start = time.time()
while rval:

    cv2.imshow("preview", frame)
    rval, frame = vc.read()

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_image, (im_cols, im_rows))

    cv2.imshow("grayscale", resized)

    x = np.resize(resized, (1, im_cols, im_rows, 1))

    predictions = np.zeros(cats)
    for i in range(nb_models):
         softmax = models[i].predict(x)
         predictions[np.argmax(softmax)] += 1

    category = np.argmax(predictions)

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
