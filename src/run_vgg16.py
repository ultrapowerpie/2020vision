import sys

import cv2
import numpy as np

from vgg16 import VGG_16

def main(args):
    f_name = args[0]
    # read in image
    im = cv2.resize(cv2.imread(f_name), (224, 224)).astype(np.float32)
    # normalize based on VGG16 train data (from authors)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = VGG_16('../vgg16_weights.h5')
    out = model.predict(im)
    print np.argmax(out) # index to VGG16 dataset categories



if __name__ == '__main__':
    main(sys.argv[1:])
