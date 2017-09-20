import os
import sys
import cv2
import pickle
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from tqdm import *

from modules.tresholds import *
from modules.warp import *
from modules.slidewindow import *
from modules.curvature import *
from modules.drawonroad import *
from modules.lanetrack import *
from modules.sanity import *

# Getting the calibration points:
with open('calib.pickle','rb') as f:
    mtx, dist = pickle.load(f)

def apply_pipeline(img):
    global laneTrack

    ksize = 7
    font = cv2.FONT_HERSHEY_SIMPLEX

    udst = cv2.undistort(img, mtx, dist, None, mtx)

    grady = abs_sobel_thresh(udst, 'y', 40, 240)
    gradx = abs_sobel_thresh(udst, 'x', 20, 200)
    mag_binary = mag_thresh(udst, ksize, (90, 220))
    dir_binary = dir_threshold(udst, sobel_kernel=7, thresh=(0.7,1.11))
    hls_binary = hls_select(udst, thresh=(100, 255))

    combined = np.zeros_like(grady)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1)] = 1
    combined = combined*255

    binary_warped, Minv = warp(combined)

    slide_img, left_fitx, right_fitx, ploty, laneTrack = slidewindow(binary_warped, laneTrack, "")

    sanity_passed = laneTrack.detected

    if (not sanity_passed):
        # We do another sidewindow with laneTrack.detected false
        slide_img, left_fitx, right_fitx, ploty, laneTrack = slidewindow(binary_warped, laneTrack, "")

    onroad = drawonroad(combined, udst, img, ploty, left_fitx, right_fitx, Minv)

    cv2.putText(onroad, 'Lane width {}'.format(laneTrack.min_distance), (230, 100),
                        font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    left_curverad, right_curverad, car_pos = curvature("", left_fitx, right_fitx, ploty, visual_on=False)

    cv2.putText(onroad, 'Left {:0.2f} m,  Right {:0.2f} m'.format(left_curverad, right_curverad), (230, 50),
                        font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(onroad, 'Deviation from center {:0.2f} m'.format(car_pos), (230, 75),
                        font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    if (sanity_passed):
        cv2.putText(onroad, 'Lanes detected', (230, 25),
                            font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(onroad, 'Detection failed', (230, 25),
                            font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    return onroad

laneTrack = LaneTrack()

#clip = VideoFileClip("project_video.mp4").subclip(25,35)
clip = VideoFileClip("project_video.mp4")
processed_clip = clip.fl_image(apply_pipeline)
processed_clip.write_videofile("project_processed_video.mp4", audio=False)
