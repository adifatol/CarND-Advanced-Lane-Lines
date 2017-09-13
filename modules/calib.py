import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import *

images = glob.glob('camera_cal/calibration*.jpg')

objpoints = []
imgpoints = []

objp       = np.zeros((9*6,3),np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

print('Calibrating ...')

for i in tqdm(range(len(images))):
    img_name = images[i]
    img      = mpimg.imread(img_name)
    gray     = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    res      = cv2.imwrite("output_images/calib/gray.jpg", gray)

    ret,corners = cv2.findChessboardCorners(gray,(9,6),None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        with_corners          = cv2.drawChessboardCorners(img,(9,6),corners,ret)
        #Saving images with drawn corners
        img_name_only         = os.path.basename(img_name)
        img_name_with_corners = 'output_images/calib/drawChessboard/{0}'.format(img_name_only)
        res = cv2.imwrite(img_name_with_corners, with_corners)

print('Undistort ...')

def cal_undistort(img, objpoints, imgpoints):
    img_size = img.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

for i in tqdm(range(len(images))):
    img_name = images[i]
    img      = mpimg.imread(img_name)
    udst     = cal_undistort(img, objpoints, imgpoints)
    img_name_only = os.path.basename(img_name)
    img_name_udst = 'output_images/calib/undistorted/{0}'.format(img_name_only)
    res = cv2.imwrite(img_name_udst, udst)
