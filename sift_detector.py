import numpy as np
from scipy import ndimage
from PIL import Image

# Implementation of sift keypoint detector
def sift_detector(img, threshold):
    h, w = img.shape
    original = np.array(img, dtype='float32')
    s = 3
    k = 2**(1.0/s)
    all_keys = []
    k_sigma = np.array([1.6, 1.6 * k, 1.6 * (k ** 2), 1.6 * (k ** 3)])
    # get the key points from four octaves
    for i in range(0, 4):
        # guassian smoothing
        octave_smooth = gaussian_blur(original, 
        		h / 2 ** i, 
        		w / 2 ** i, 
        		2 ** i * k_sigma)
        # calculate the Difference of Gaussian
        octave_dog = dog(h / 2 ** i, w / 2 ** i, octave_smooth)
        # extract the key points
        all_keys.extend(key_extract(octave_dog, threshold, i))

    return all_keys

# Gaussian smoothing the picture
def gaussian_blur(img, h, w, sigma):
    pyr_level = np.zeros((h, w, 4))
    for i in range (0, 4):
        temp = ndimage.filters.gaussian_filter(img, sigma[i])
        pyr_level[:,:,i] = np.array(Image.fromarray(temp).resize((w, h)))
    
    return pyr_level

# Calculate the difference of Gaussian
def dog(h, w, pyr_level):
    dog_level = np.zeros((h, w, 3))
    for i in range(0, 3):
        dog_level[:, :, i] = np.absolute(
        	pyr_level[:, :, i + 1] - pyr_level[:, :, i])

    return dog_level

# Extract the extremas
def key_extract(octave, threshold, order):
    delta = 1e-3
    h, w, d = octave.shape
    assert d == 3, "Unexpected dog shape: {}".format(octave)
    octave_keys = []

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if np.absolute(octave[i, j, 1]) < threshold:
                continue

            localMax = octave[i - 1:i +  1, j - 1:j + 1, :].max()
            isKey = np.absolute(octave[i, j, 1] - localMax) < delta
            if not isKey:
                continue

            dx = (octave[i + 1, j, 1] - 
            	octave[i - 1, j, 1]) * 0.5 / 255
            dy = (octave[i, j + 1, 1] - 
            	octave[i, j - 1, 1]) * 0.5 / 255
            ds = (octave[i, j, 2] - 
            	octave[i, j, 0]) * 0.5 / 255

            dxx = (octave[i + 1, j, 1] + 
            	octave[i - 1, j, 1] - 
            	2 * octave[i, j, 1]) / 255
            dyy = (octave[i, j + 1, 1] + 
            	octave[i, j - 1, 1] - 
            	2 * octave[i, j, 1]) / 255
            dss = (octave[i, j, 2] + 
            	octave[i, j, 0] - 
            	2 * octave[i, j, 1]) / 255
            dxy = (
                octave[i + 1, j + 1, 1] - 
                octave[i + 1, j - 1, 1] - 
                octave[i - 1, j + 1, 1] + 
                octave[i - 1, j - 1, 1]
            ) * 0.25 / 255
            dxs = (
                octave[i + 1, j, 2] - 
                octave[i + 1, j, 0] - 
                octave[i - 1, j, 2] + 
                octave[i - 1, j, 0]
            ) * 0.25 / 255
            dys = (
                octave[i, j + 1, 2] - 
                octave[i, j - 1, 2] - 
                octave[i, j + 1, 0] + 
                octave[i, j - 1, 0]
            ) * 0.25 / 255

            dD = np.matrix([[dx], [dy], [ds]])
            H = np.matrix([
                [dxx, dxy, dxs], 
                [dxy, dyy, dys], 
                [dxs, dys, dss]]
            )
            x_hat = np.linalg.lstsq(H, dD, rcond=None)[0]
            D_x_hat = octave[i][j][1] + 0.5 * np.dot(dD.transpose(), x_hat)

            r = 10.0
            # Discard the keypoints D(x_hat) < 0.03 and the edge points
            if (((dxx + dyy) ** 2) * r < ((
            	dxx * dyy - (dxy ** 2)) * (((r + 1) ** 2)))) and (
                np.absolute(D_x_hat) > 0.03):
                octave_keys.append([j * (2 ** order), 
                		i * (2 ** order), 
                		octave[i, j, 1]])

    return octave_keys
