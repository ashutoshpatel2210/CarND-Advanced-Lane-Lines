## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:
* Packed used during project
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/Undistort_image.png "Undistorted"
[image2]: ./output_images/Undistort_output1.png "Undistorted test images"
[image3]: ./output_images/Undistort_output2.png "Undistorted test images"
[image4]: ./output_images/Undistort_output3.png "Undistorted test images"
[image5]: ./output_images/Undistort_output4.png "Undistorted test images"
[image6]: ./output_images/Undistort_output5.png "Undistorted test images"
[image7]: ./output_images/Undistort_output6.png "Undistorted test images"
[image8]: ./output_images/Undistort_output7.png "Undistorted test images"
[image9]: ./output_images/Undistort_output8.png "Undistorted test images"
[image10]: ./output_images/Binary_output1.png "Binary output of test images"
[image11]: ./output_images/Binary_output2.png "Binary output of test images"
[image12]: ./output_images/Binary_output3.png "Binary output of test images"
[image13]: ./output_images/Binary_output4.png "Binary output of test images"
[image14]: ./output_images/Binary_output5.png "Binary output of test images"
[image15]: ./output_images/Binary_output6.png "Binary output of test images"
[image16]: ./output_images/Binary_output7.png "Binary output of test images"
[image17]: ./output_images/Binary_output8.png "Binary output of test images"

[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

### Import and initialize the packages needed in the project

I imported packages needed in the project

* OpenCV - an open source computer vision library,
* Matplotbib - a python 2D plotting libray,
* Numpy - a package for scientific computing with Python,
* MoviePy - a Python module for video editing.
* os -  Miscellaneous operating system interfaces
* glob — Unix style pathname pattern expansion

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second and third code cell of the IPython notebook located in "AdvancedLaneFinding.ipynb". 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction distortion correction to the test image using the `cv2.undistort()` function and obtained this result of the all test images like this one:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image in cell 5 and output of the result is disaplyed in cell 6. I used red and green channel color threshold(threshold> 200) to detect yellow line properly in all conditions (shadow or bright light conditions), I also used HLS color thresold (Lightness and Value represent different ways to measure the relative lightness or darkness of a color), These min and max thresolds are as follows:  L channel (130,255)and S channel threshold (150, 255). I also applied gradient x threshold to channel s to detect edges in image. So binary image is combination of R & G, L, S colors and gradient in x driections. Here are the output of test images after applying the above procedure. 

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

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
