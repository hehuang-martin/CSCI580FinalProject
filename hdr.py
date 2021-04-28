from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
import argparse
import os
import glob

def loadExposureSeq(path):
    images = []
    times = []
    with open(os.path.join(path, 'list.txt')) as f:
        content = f.readlines()
    for line in content:
        tokens = line.split()
        images.append(cv2.imread(os.path.join(path, tokens[0])))
        times.append(1 / float(tokens[1]))

    return images, np.asarray(times, dtype=np.float32)

image_path = './image/test1/'
exposure_file = './image/test1.txt'

## [Load images and exposure times]
# load the images
image_files = glob.glob(image_path + "*")
images = []
for file in image_files:
    image = cv2.imread(file)
    image = cv2.resize(image, (960, 540))
    images.append(image)
times = []
file = open(exposure_file, 'r')
for x in file:
    times.append(float(x))
times = np.asarray(times, dtype = np.float32)
## [Load images and exposure times]

## [Estimate camera response]
calibrate = cv2.createCalibrateDebevec()
response = calibrate.process(images, times)
## [Estimate camera response]

## [Make HDR image]
merge_debevec = cv2.createMergeDebevec()
hdr = merge_debevec.process(images, times, response)
## [Make HDR image]

## [Tonemap HDR image]
tonemap = cv2.createTonemap(2.2)
ldr = tonemap.process(hdr)
## [Tonemap HDR image]

## [Perform exposure fusion]
merge_mertens = cv2.createMergeMertens()
fusion = merge_mertens.process(images)
## [Perform exposure fusion]

## [Write results]
cv2.imwrite('fusion.png', fusion * 255)
cv2.imwrite('ldr.png', ldr * 255)
cv2.imwrite('hdr.png', hdr)
## [Write results]

