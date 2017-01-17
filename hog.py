#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 12:00:47 2017

@author: Kush

"""

import cv2
import os
import numpy as np

def main():

    winSize = (64, 128)     # window size, align to block size and stride
    blockSize = (16,16)     # 2x2 cells - used for normalization, align to cell size
    blockStride = (8,8)     # must be a multiple of cell size
    cellSize = (8,8)
    nbins = 9               # number of bins in histogram
    derivAperture = 1       # don't know what this is really
    winSigma = 4.           # gaussian smoothing window parameter
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0     # flag to specify gamma correction preprocessing
    nlevels = 64            # maximum number of detection window increases
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    
    
    #compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),) # not sure about these parameters
        
    # read images from folder
    trials = 1250
    images = []
    
    
    testtrials = 500         # number of images to take from each class, max 2k
    testimages = []
    
    cumulative = trials + testtrials
    
    # take 200 uimages from each class to train SVM
    for x in range(0, 10):
        count = 0
        # change folder depending on directory
        folder = "/Users/Kush/python/hog/imgs/train/c" + str(x)
        for fn in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,fn))
            if count < trials:
                if img is not None:
                    images.append(img)
                    count += 1
            else: #test images loaded
                testimages.append(img)
                count += 1
                if (count > (cumulative - 1)):
                    break
    
    length = len(images)
    
    hogmatrix = np.float32(np.zeros(shape=(length,3780)))
    count = 0
    for img in images:
        hist = hog.compute(img,winStride,padding,locations)
        hogmatrix[count] = np.float32(hist).reshape(-1,3780)
        count += 1
    
    responses = np.repeat([np.arange(10)],trials)
   
    svm = cv2.SVM()
    
    # default params from OpenCV
    svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )
   
    svm.train(hogmatrix,responses, params=svm_params)
#    svm.train_auto(trainData, responses) # train using optimal parameters

    length = len(testimages)
    
    testmatrix = np.float32(np.zeros(shape=(length,3780)))
    count = 0
    for img in testimages:
        hist = hog.compute(img,winStride,padding,locations)
        testmatrix[count] = np.float32(hist).reshape(-1,3780)
        count += 1
    
    result = svm.predict_all(testmatrix)
    truth = np.repeat([np.arange(10)],testtrials).reshape(testtrials*10,1)
    
    # check accuracy - code taken from tutorial
    accuracy = (result==truth)
    correct = np.count_nonzero(accuracy)
    print (correct*100.0/result.size)
    
    # main program starts here
main()