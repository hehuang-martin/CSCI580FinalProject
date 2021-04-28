from typing import final
import numpy as np
import cv2
import pandas as pd
import os
import argparse
import glob
import random
import math
import matplotlib.pyplot as plt

from hdrtool import hdr # test purpose


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


def expandGrid(array, n):
    res = np.zeros(array.shape + (n, ))
    axe_range = []
    for i in range(array.ndim):
        axe_range.append(range(array.shape[i]))
    f_grid = []
    for i in np.meshgrid(*axe_range, indexing='ij'):
        f_grid.append(i.ravel())
    res[tuple(f_grid + [array.ravel()])] = 1
    assert((res.sum(-1) == 1).all())
    assert(np.allclose(np.argmax(res, -1), array))
    return res


def computeResponseCurve(color, exposures, smoothing):
    color_range = 256
    num_of_pixel = color.shape[0]
    num_of_image = color.shape[1]

    A = np.zeros((num_of_image * num_of_pixel + color_range + 1, color_range + num_of_pixel))
    B = np.zeros((A.shape[0], 1))

    # including data fitting equation
    k = 0
    for i in range(num_of_pixel):
        for j in range(num_of_image):
            c_ij = color[i, j]
            w_ij = weightFunction(c_ij)
            A[k, c_ij] = w_ij
            A[k, (color_range) + i] = -w_ij
            B[k] = w_ij * exposures[j]
            k += 1

    # fix the curve by setting middle value to one
    A[k, math.floor(color_range / 2.)] = 1
    k += 1

    # apply smoothing
    for i in range(1, color_range - 2):
        w_i = weightFunction(i + 1)
        A[k, i] = w_i * smoothing
        A[k, i + 1] = -2 * w_i * smoothing
        A[k, i + 2] = w_i * smoothing
        k += 1

    # solving it using Singular Value Decomposition
    x = np.linalg.lstsq(A, B, rcond=-1)
    x = x[0]
    g = x[0: color_range]
    le = x[color_range, :]
    
    return g[:, 0]


def computeRadianceMap(images, exposure, response_curve):
    zmid = (255-0)/2
    imgarr = np.asarray(images) #(P photos, Length, Width)
    img_shape = images[0].shape
    img_rad_map = np.zeros(img_shape, dtype=np.float64)

    num_of_images = len(images)
    H, W = images[0].shape
    w = np.zeros((num_of_images, H, W))
    for i in range(num_of_images):
        for j in range(H):
            for k in range(W):
                w[i, j, k] = weightFunction(w[i, j, k])
    G = np.dot(expandGrid(imgarr, 256),response_curve)
    W = zmid - np.absolute(imgarr-zmid)
    
    lndT = exposure[:, np.newaxis, np.newaxis] * np.ones(imgarr.shape)
    img_rad_map[:, :] = np.sum((w * (G - lndT)), axis=0) / (np.sum(w, axis=0) + 1e-8)

    return np.exp(img_rad_map)


# implement debevec algorithm for getting hdr image
def getHDR(images, exposure_times, l):
    rgb_channels = images[0].shape[2]
    hdr_image = np.zeros(images[0].shape, dtype=np.float64)
    num_of_images = len(images)

    # new added
    num_of_samples = math.ceil(255 * 2 / (num_of_images - 1)) * 2
    random_indexes = np.random.choice(images[0].shape[0] * images[0].shape[1], (num_of_samples,), replace=False)

    rgb = ['r', 'g', 'b']
    for i in range(rgb_channels):
        print("Color Channel: " + rgb[i])
        # get the single color from each image
        # size of [image.height, image.weight, 1]
        single_color = []
        for image in images:
            single_color.append(image[:, :, i])

        # anohter way doing sample
        # sampling the color
        this_color = np.zeros((num_of_samples, num_of_images), dtype=np.uint8)
        for j, image in enumerate(images):
            flat_image = image[:, :, i].flatten()
            this_color[:, j] = flat_image[random_indexes]

        # compute response curve
        print(" Compute Reponse Cuve ... ")
        response_curve = computeResponseCurve(this_color, exposure_times, l)
        response_curves.append(response_curve)

        # compute radiance map
        print(" Compute Radiance Map ... ")
        img_rad_map = computeRadianceMap(single_color, exposure_times, response_curve)
        hdr_image[..., i] = img_rad_map

    return hdr_image


def tone_mapping(image, max_white, alpha):
    # global tone mapping
    delta = 1e-9
    n = image.shape[0] * image.shape[1] * 3
    lw = image
    lb = np.exp((1/n) * np.sum(np.log(delta + lw)))
    lm = (alpha/lb) * lw
    ld = lm * (1 + (lm / (max_white**2))) / (1 + lm)
    mapped_image = ld
    mapped_image = ((mapped_image - mapped_image.min()) * (255 / (mapped_image.max() - mapped_image.min())))
    return mapped_image

# load the image and exposure time
if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', default='./image/test1/', type=str,
                        help='directory of the folder containing images')
    parser.add_argument('--exposure-file', default='./image/test1.txt', type=str,
                        help='file containing the exposure time for each image')
    parser.add_argument('--output-path', default='', type=str,
                        help='output path of jpg file')
    parser.add_argument('--lwhite', type=float, default=0.8,
                        help='the number for constraint the highest value in hdr image(default: 0.8)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='The number for correction. Higher value for brighter result; lower for darker(default: 0.5)')

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
        image = cv2.resize(image, (960, 540))
        images.append(image)
    image_height = images[0].shape[0]
    image_width = images[0].shape[1]
    print("Pixel Number: " + str(image_height * image_width))

    # load exposure times
    exposure_times = []
    file = open(exposure_file, 'r')
    for x in file:
        exposure_times.append(float(x))
    exposure_times = np.asarray(exposure_times)

    # compute hdr image
    print("Calculating HDR image ... ")
    hdr_image = getHDR(images, np.log(exposure_times).astype(np.float32), 100)
    print("Got HDR Image ... ")
    cv2.imwrite(output_path + "radiance_map.jpg", hdr_image)

    # tone mapping
    print(" Tone Mapping ... ")
    final_image = tone_mapping(hdr_image, max_white=np.exp(hdr_image.max())*0.7, alpha=0.5)

    # save the imagee
    cv2.imwrite(output_path + output_file, final_image.astype(np.uint8))

    # plot response curve
    print(" Saving Response Curve ")
    print(" Plotting Response Curve ")
    px = list(range(0, 256))
    plt.figure(constrained_layout=False, figsize=(5, 5))
    plt.title("Response curves for BGR", fontsize=20)
    plt.plot(px, np.exp(response_curves[0]), 'r')
    plt.plot(px, np.exp(response_curves[1]), 'b')
    plt.plot(px, np.exp(response_curves[2]), 'g')
    plt.ylabel("log Exposure X", fontsize=20)
    plt.xlabel("Pixel value Z", fontsize=20)
    plt.savefig(output_path+"rc.png")