import cv2
import numpy as np

def warp(img):
    src = np.float32([[(200, 720), (570, 470), (720, 470), (1130, 720)]])
    dst = np.float32([[(350, 720), (350, 0), (980, 0), (980, 720)]])

    ## Grab the image shape
    img_size = (img.shape[1], img.shape[0])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    # warped = cv2.warpPerspective(img, M, dsize = img.shape, flags = cv2.INTER_LINEAR)

    # src = np.array([(178, 660), (550, 441), (720, 441), (1130, 660)], dtype=np.int32)
    # dst = np.array([(350, 660), (300, 0), (1030, 0), (980, 660)], dtype=np.int32)
    #
    # unwarped = cv2.polylines(img, [src],True,(255,0,0), 5)
    # warped   = cv2.polylines(warped, [dst],True,(0,0,255), 5)

    return warped, Minv
