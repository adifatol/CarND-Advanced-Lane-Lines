import os
import sys
import cv2
import pickle
import glob
import matplotlib.image as mpimg

from tqdm import *

from modules.tresholds import *
from modules.warp import *
from modules.slidewindow import *
from modules.curvature import *
from modules.drawonroad import *

#Use -c argument to run calibration
from sys import argv
if '-c' in argv:
    import modules.calib # Run calibration and save points

# Getting back the calibration points:
with open('calib.pickle','rb') as f:
    mtx, dist = pickle.load(f)

test_images = glob.glob('test_images/test*.jpg')

def save_img(prepath,img_name,img):
    img_name_only = os.path.basename(img_name)
    img_name_full = '{0}{1}'.format(prepath,img_name_only)
    # img_to_save   = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    res = cv2.imwrite(img_name_full, img)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

for i in tqdm(range(len(test_images))):
    img_name = test_images[i]
    img      = mpimg.imread(img_name)
    udst     = cv2.undistort(img, mtx, dist, None, mtx)

    save_img('output_images/test_images/undistorted/',img_name,udst)

    ksize = 7

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(udst, 'x', 20, 200)
    grady = abs_sobel_thresh(udst, 'y', 40, 240)
    mag_binary = mag_thresh(udst, ksize, (90, 220))
    dir_binary = dir_threshold(udst, sobel_kernel=7, thresh=(0.7,1.11))
    hls_binary = hls_select(udst, thresh=(100, 255))

    combined = np.zeros_like(grady)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1)] = 1

    save_img('output_images/test_images/comb_tresholds/gradx/',img_name,gradx*255)
    save_img('output_images/test_images/comb_tresholds/grady/',img_name,grady*255)
    save_img('output_images/test_images/comb_tresholds/mag/',img_name,mag_binary*255)
    save_img('output_images/test_images/comb_tresholds/dir/',img_name,dir_binary*255)
    save_img('output_images/test_images/comb_tresholds/hls/',img_name,hls_binary*255)
    save_img('output_images/test_images/comb_tresholds/combined/',img_name,combined*255)

    combined = combined*255
    binary_warped, Minv = warp(combined)
    save_img('output_images/test_images/warped/',img_name,binary_warped)

    slide_img, left_fitx, right_fitx, ploty = slidewindow(binary_warped, img_name)
    save_img('output_images/test_images/slidewindow/',img_name,slide_img)

    curvature(img_name, left_fitx, right_fitx, ploty)

    onroad = drawonroad(combined, udst, img, ploty, left_fitx, right_fitx, Minv)
    save_img('output_images/test_images/onroad/',img_name,onroad)
