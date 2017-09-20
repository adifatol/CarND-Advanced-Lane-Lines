import cv2
import os
import numpy as np
from modules.sanity import *

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def slidewindow(binary_warped, laneTrack, img_name=""):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    avg_frames = 10 # number of frames to average
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    if (laneTrack.detected):
        leftx_base = laneTrack.leftLane.bestx
        rightx_base = laneTrack.rightLane.bestx
    else:
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    laneTrack.detected = True

    if (len(lefty) < 1 or
        len(leftx) < 1 or
        len(righty) < 1 or
        len(rightx) < 1):
        laneTrack.detected = False
        # Fit a second order polynomial to each
        left_fit = laneTrack.leftLane.current_fit
        right_fit = laneTrack.rightLane.current_fit
        laneTrack.detected = False
    else :
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    #Check if lanes are parallel, used for sanity check
    parallel_diff = np.abs(np.subtract(left_fit, right_fit))
    #Minimal distance between the lanes
    diff = np.subtract(right_fitx, left_fitx)*xm_per_pix
    min_distance = np.amin(diff)
    laneTrack.parallel_diff = parallel_diff
    laneTrack.min_distance  = min_distance

    sanity_passed = sanity(laneTrack)
    if (sanity_passed):
        laneTrack.leftLane.bestx = leftx_current
        laneTrack.rightLane.bestx = rightx_current
        laneTrack.leftLane.current_fit = left_fit
        if (len(laneTrack.rightLane.allx) > 0):
            laneTrack.rightLane.allx = (right_fitx + laneTrack.rightLane.allx*avg_frames) / (avg_frames+1)
        else:
            laneTrack.rightLane.allx = right_fitx
        if (len(laneTrack.leftLane.allx) > 0):
            laneTrack.leftLane.allx = (left_fitx + laneTrack.leftLane.allx*avg_frames) / (avg_frames+1)
        else:
            laneTrack.leftLane.allx = left_fitx
        laneTrack.leftLane.ally = ploty
        laneTrack.rightLane.ally = ploty
    else:
        #we gonna use the old values
        left_fitx = laneTrack.leftLane.allx
        right_fitx = laneTrack.rightLane.allx
        ploty = laneTrack.rightLane.ally

    if (len(laneTrack.rightLane.allx) > 0):
        right_fitx = (right_fitx + laneTrack.rightLane.allx*avg_frames) / (avg_frames+1)
    if (len(laneTrack.leftLane.allx) > 0):
        left_fitx = laneTrack.leftLane.allx = (left_fitx + laneTrack.leftLane.allx*avg_frames) / (avg_frames+1)

    return out_img, left_fitx, right_fitx, ploty, laneTrack
