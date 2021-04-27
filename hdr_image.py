import numpy as np
import cv2
import pandas as pd
import os
import argparse
import glob
import random
import math
import matplotlib.pyplot as plt

from hdrtool import hdr

output_path = ''
output_file = 'output.jpg'
response_curves = []

# the weight function for single pixel
def weightFunction(pixel):
    color_min = 0
    color_max = 255
    color_mid = math.floor((color_min + color_max) / 2)

    if pixel <= color_mid:
        pixel = pixel - color_min + 1
    else:
        pixel = color_max - pixel + 1

    return pixel

def computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, weighting_function):
    color_range = 256
    n = intensity_samples.shape[0] # Numbers of pixels we picked    s1
    P = intensity_samples.shape[1] # Numbers of images we take in different exposures   s2
    
    # NxP + [(Zmax-1) - (Zmin + 1)] + 1 constraints; N + 256 columns
    mat_A = np.zeros((P * n + color_range + 1, color_range + n))
    mat_b = np.zeros((mat_A.shape[0], 1))

    # including data fitting equation
    k = 0
    for i in range(n):
        for j in range(P):
            z_ij = intensity_samples[i, j]
            w_ij = weighting_function(z_ij)
            mat_A[k, z_ij] = w_ij
            mat_A[k, (color_range) + i] = -w_ij
            mat_b[k] = w_ij * log_exposures[j]
            k += 1
    
    # fix the curve by setting middle value to zero
    mat_A[k, math.floor(color_range / 2.)] = 1
    k += 1

    # apply smoothing
    for i in range(1, color_range - 2):
        w_i = weightFunction(i + 1)
        mat_A[k, i] = w_i * smoothing_lambda
        mat_A[k, i+1] = -2 * w_i * smoothing_lambda
        mat_A[k, i+2] = w_i * smoothing_lambda
        k += 1
    
    # solving it using Singular Value Decomposition
    x = np.linalg.lstsq(mat_A, mat_b, rcond = -1)
    x = x[0]
    g = x[0: color_range]

    return g[:, 0]



# implement debevec algorithm for getting hdr image
def getHDR(images, exposure_times, l):
    rgb_channels = images[0].shape[2]
    hdr_image = np.zeros(images[0].shape, dtype=np.float64)
    num_of_images = len(images)

    # new added
    num_of_samples = math.ceil(255*2/(num_of_images-1))*2
    random_indexes = np.random.choice(images[0].shape[0] * images[0].shape[1], (num_of_samples,), replace=False)


    for i in range(rgb_channels):
        # get the single color from each image
        # size of [image.height, image.weight, 1]
        single_color = []
        for image in images:
            single_color.append(image[:, :, i])

        # anohter way doing sample

        
        # sampling the color
        this_color = np.zeros((num_of_samples, num_of_images), dtype = np.uint8)
        for j, image in enumerate(images):
            flat_image = image[:,:,i].flatten()
            this_color[:, j] = flat_image[random_indexes]


        # fukcing idiot way of sampling
        color_sampled = np.zeros((num_of_samples, num_of_images), dtype=np.uint8)
        middle_image_index = math.floor(num_of_images / 2.)
        middle_image_color = single_color[middle_image_index]
        
        for j in range(num_of_samples):
            rows, cols = np.where(middle_image_color == j)
            if len(rows) > 0:
                index = random.randrange(len(rows))
                index2 = random.randrange(len(cols))
                for k in range(len(images)):
                    color_sampled[j][k] = single_color[k][rows[index], cols[index2]]

        
        # compute response curve
        response_curve = computeResponseCurve(this_color, exposure_times, l, weightFunction)
        response_curves.append(response_curve)

        #img_rad_map = hdr.computeRadianceMap(single_color, exposure_times, response_curve, weightFunction)
        #hdr_image[..., i]  = img_rad_map

    return np.exp(hdr_image)


# load the image and exposure time
if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', default='./image/test1/',  type=str,
                        help = 'directory of the folder containing images')
    parser.add_argument('--exposure-file', default = './image/test1.txt', type=str,
                        help = 'file containing the exposure time for each image')
    parser.add_argument('--output-path', default= '', type = str,
                        help = 'output path of jpg file')
    parser.add_argument('--lwhite', type = float, default = 0.8,
                        help = 'the number for constraint the highest value in hdr image(default: 0.8)')
    parser.add_argument('--alpha', type = float, default = 0.5,
                        help = 'The number for correction. Higher value for brighter result; lower for darker(default: 0.5)')


    args = parser.parse_args()
    image_path = args.image_path
    exposure_file = args.exposure_file
    output_path = args.output_path
    lwhite = args.lwhite
    alpha = args.alpha

    # load the images
    image_files = glob.glob(image_path + "*")
    images = []
    for file in image_files:
        image = cv2.imread(file)
        images.append(image)

    # load exposure times
    exposure_times = []
    file = open(exposure_file, 'r')
    for x in file:
        exposure_times.append(float(x))
    exposure_times = np.asarray(exposure_times)

    
    # compute hdr image
    hdr_image = getHDR(images, np.log(exposure_times).astype(np.float32), 100)

    # tone mapping
    #final_image = hdr.globalToneMapping(hdr_image, Lwhite=np.exp(hdr_image.max())*0.8, alpha=0.5)

    # save the imagee
    #cv2.imwrite(output_path + output_file, final_image.astype(np.uint8))

    # plot response curve
    print(" Saving Response Curve ")
    print(" Plotting Response Curve ")
    px = list(range(0,256))
    plt.figure(constrained_layout=False,figsize=(5,5))
    plt.title("Response curves for BGR", fontsize=20)
    plt.plot(px,np.exp(response_curves[0]),'r')
    plt.plot(px,np.exp(response_curves[1]),'b')
    plt.plot(px,np.exp(response_curves[2]),'g')
    plt.ylabel("log Exposure X", fontsize=20)
    plt.xlabel("Pixel value Z", fontsize=20)
    plt.show()
    