**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./test_images/test1.jpg "Original"
[image2]: ./output_images/test_images/undistorted/test1.jpg "Undistorted"
[image3]: ./output_images/test_images/comb_tresholds/combined/test1.jpg "Combined Treshold"
[image4]: ./output_images/test_images/unwarped/test1.jpg "Warp Example Src Points"
[image5]: ./output_images/test_images/warped/test1.jpg "Warp Example"
[image6]: ./output_images/test_images/histograms/test7.jpg "Histogram"
[image7]: ./output_images/test_images/curvature/test7.jpg "Polynomial and curvature"

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/adifatol/CarND-Advanced-Lane-Lines/blob/master/writeup.md) is the writeup for this project.

You're reading it!

### Project Structure

For this project I used python scripts only. I tried to implement the code as modular as possible so for each step in the pipeline there is a [module](https://github.com/adifatol/CarND-Advanced-Lane-Lines/tree/master/modules).

I first implemented a [pipeline](https://github.com/adifatol/CarND-Advanced-Lane-Lines/blob/master/pipeline.py) for the test images using the modules for [calibration](https://github.com/adifatol/CarND-Advanced-Lane-Lines/blob/master/modules/calib.py), [tresholding](https://github.com/adifatol/CarND-Advanced-Lane-Lines/blob/master/modules/tresholds.py), [image warping](https://github.com/adifatol/CarND-Advanced-Lane-Lines/blob/master/modules/warp.py) etc.

The second step was to re-use the same modules in the [video pipeline](https://github.com/adifatol/CarND-Advanced-Lane-Lines/blob/master/pipeline_video.py) script. The modules were slightly modified in order to work better on the entire video but with some small changes they can be refitted to work on the images pipeline again.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the [calibration module](https://github.com/adifatol/CarND-Advanced-Lane-Lines/blob/master/modules/calib.py).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

<img src="https://github.com/adifatol/CarND-Advanced-Lane-Lines/blob/master/camera_cal/calibration2.jpg" width="250"> <img src="https://github.com/adifatol/CarND-Advanced-Lane-Lines/blob/master/output_images/calib/drawChessboard/calibration2.jpg" width="250"> <img src="https://github.com/adifatol/CarND-Advanced-Lane-Lines/blob/master/output_images/calib/undistorted/calibration2.jpg" width="250">

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image1]

The pipeline.py checks if the "-c" command line argument is passed. If this is true, the ["calib.py"](https://github.com/adifatol/CarND-Advanced-Lane-Lines/blob/master/modules/calib.py) module is imported. This module applies the algorithm explained in the previous point in order to find the camera calibration parameters. These parameters are then saved in the [calib.pickle](https://github.com/adifatol/CarND-Advanced-Lane-Lines/blob/master/calib.pickle) file.

After this (independednt of the "-c" argument), the pipeline will load these parameters into mtx and dist variables. The cv2.undistort() function is then applied using these calibration parameters giving the folowing result:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

After the image was undistorted, I applied a combination of color and gradient [thresholds](https://github.com/adifatol/CarND-Advanced-Lane-Lines/blob/master/modules/tresholds.py) to generate a binary image (lines 47 through 54 in [pipeline.py](https://github.com/adifatol/CarND-Advanced-Lane-Lines/blob/master/pipeline.py)). For the video pipeline the hls_select function was modified in order to include a color treshold on the L channel. Initially the function applied color threshold for the S channel only, which gave good results for the test images. The treshold on the L channel greatly improved the results on the video in the shadow areas especially. 

Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which is found in the [warp module](https://github.com/adifatol/CarND-Advanced-Lane-Lines/blob/master/modules/warp.py).  The `warp()` function takes as input an image (`combined`), which is the result of the combined tresholds from the above point.  I chose to hardcode the source and destination points by intuition and trial and error and found that the following results work well enough:

```python
    src = np.float32([[(200, 720), (570, 470), (720, 470), (1130, 720)]])
    dst = np.float32([[(350, 720), (350, 0), (980, 0), (980, 720)]])
```

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 350, 720      | 
| 570, 470      | 350, 0        |
| 720, 470      | 980, 720      |
| 1130, 720     | 980, 0        |

Here's an example of trying to find good source points:
![alt text][image4]
Here is a warped image result (applied on combined tresholds):
![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the next step I applied the ["sliding window"](https://github.com/adifatol/CarND-Advanced-Lane-Lines/blob/master/modules/slidewindow.py) method in order to find the lane lines.
Initially, the sliding window algorithm starts from the points found running a histogram function in order to find the highest peaks on the binary warped image:
![alt text][image6]

Then I fit my lane lines with a 2nd order polynomial and plotted the result on an image (this example includes curvature radius calculations also, explained in the next point):

![alt text][image7]

When used on the video pipeline, the sliding window is running some checks if the previous lanes were detected correctly. If yes, the starting points are not chosen from the histogram but actually from the points that were detected in the previous frame. If the lanes are correctly detected, then the pixels calculated by the polyfit are averaged over the last 10 frames and used as the new found lanes. If the lanes are not detected correctly (checking the distance between lanes to be about ~3m), the previous lanes detected are used (from the previous frame).

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature was calculated in the [curvature module](https://github.com/adifatol/CarND-Advanced-Lane-Lines/blob/master/modules/curvature.py). The deviation from the center of the lanes is calculated in the same module.
Here is an example with the curvature values calculated and printed on the warped image:
![alt text][image7]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
