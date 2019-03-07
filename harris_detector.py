import numpy as np
import math

#Implementation of Harris Corner Detector
def harris_detector(img, threshold):

    k = 0.04

    r = np.zeros(img.shape, dtype=int)

    corner_list = []

    sobel_x = np.array([[-1, 0, 1], 
            [-2, 0, 2], 
            [-1, 0, 1]], 
            dtype='float') / 8
    sobel_y = np.array([[-1, -2, -1], 
            [0, 0, 0], 
            [1, 2, 1]], 
            dtype='float') / 8

    dx = apply_filter(img, sobel_x)
    dy = apply_filter(img, sobel_y)

   
    Ixx = dx**2
    Ixy = dx * dy
    Iyy = dy**2
    
    h, w = img.shape
    for i in range(1, h - 1):
        for j in range (1, w - 1):
            Ixx_wind = Ixx[i - 1: i + 1 + 1, j - 1: j + 1 + 1]
            Ixy_wind = Ixy[i - 1: i + 1 + 1, j - 1: j + 1 + 1]
            Iyy_wind = Iyy[i - 1: i + 1 + 1, j - 1: j + 1 + 1]
            Sxx = Ixx_wind.sum()
            Sxy = Ixy_wind.sum()
            Syy = Iyy_wind.sum()

            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy

            r[i][j] = det - k * (trace**2)
            if (r[i][j] > threshold): 
                corner_list.append([j, i, r[i][j]]);

    return corner_list;

# Generalize the process of appling the mask on the pictures
def apply_filter(img, f):
    img_h, img_w = img.shape
    ret = np.array(img, dtype='float')
    img = np.array(img, dtype='float')
    f_h, f_w = f.shape
    assert f_h % 2 == 1, 'assume filter size is odd'
    f_size = np.int((f_h - 1) / 2)

    for i in range(img_h):
        for j in range(img_w):
            if (i - f_size < 0 or j - f_size < 0 
                or i + f_size >= img_h or j + f_size >= img_w):
                ret[i][j] = 0
                continue
            v = 0
            for di in range(-f_size, f_size + 1):
                for dj in range(-f_size, f_size + 1):
                    ci = i + di
                    cj = j + dj
                    fi = di + f_size
                    fj = dj + f_size
                    v = v + f[fi, fj] * img[ci, cj]
            ret[i][j] = v

    return ret





