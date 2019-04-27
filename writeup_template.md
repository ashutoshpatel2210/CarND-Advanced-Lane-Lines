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
[image18]: ./output_images/Wrapped_output.png "Wrapped output ot test image 3"
[image19]: ./output_images/histogram.png "Histogram of wrapped output of test image 3"
[image20]: ./output_images/sliding_window.png "Sliding window of test image 3"
[image21]: ./output_images/Similarlines.png "Detect lanes with of test image 3"
[image22]: ./output_images/Final_output1.png "Final output of test images"
[image23]: ./output_images/Final_output2.png "Final output of test images"
[image24]: ./output_images/Final_output3.png "Final output of test images"
[image25]: ./output_images/Final_output4.png "Final output of test images"
[image26]: ./output_images/Final_output5.png "Final output of test images"
[image27]: ./output_images/Final_output6.png "Final output of test images"
[image28]: ./output_images/Final_output7.png "Final output of test images"
[image29]: ./output_images/Final_output8.png "Final output of test images"

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

The code for my perspective transform includes a function called `warper()`, which appears in cell 7 and 8 in the file `AdvancedLaneFinding.ipynb` The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
            ([590,450],[687,450],[1100,720],[200,720]))
dst = np.float32(
            ([300,0],[900,0],[900,720],[300,720]))
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 590, 450      | 300, 0        | 
| 200, 720      | 300, 720      |
| 1100, 720     | 900, 720      |
| 687, 450      | 900, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image18]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to detect lane lines, first we need to define where to look for seach for lanes in image, I used histogram of x-positions to locate lanes from image. histogram will be good indicators of the x-position of the base of the lane lines. I used that as a starting point for where to search for the lines. 
![alt text][image19]

From that point, I use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame. First, i split the histogram into two sides, one for each lane line, next I setup window and hyperparameter related to sliding windows to select particular lane in image. 

```python
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
  ```
I iterate through nwindwos to look for curvature and find the mean position of activated pixels within the window to have shifted. After finding all pixels, I used functions np.polyfit(0 to fit polynomial to line, after finding polynomial, i skipped  sliding window approach to find lines withing the boundary of margin. I used only those pixels with x-values that are +/-  margin from polynomial lines. 

![alt text][image20]
![alt text][image21]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in measure_curvature_real() function my code in `AdvancedLaneFinding.ipynb`
```python
def measure_curvature_real(left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    ploty = np.linspace(0, 719, num=720)
    left_fit_cr, right_fit_cr = left_fit, right_fit
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])  ## Implement the calculation of the left line here
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])  ## Implement the calculation of the right line here
    
    return left_curverad, right_curverad
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in my code in `AdvancedLaneFinding.ipynb` in the function `final_pipeline()`.   I combined all techniques described above to detect lane in the function `final_pipeline()`. The code also disabled average curvature and centre offset in meters in image. 

An example of its output can be observed below:

![alt text][image22]
![alt text][image23]
![alt text][image24]
![alt text][image25]
![alt text][image26]
![alt text][image27]
![alt text][image28]
![alt text][image29]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

# Issues faced:
 * Video has some part in bright light condition and some part in dark light condition, The same color thresholds can not be used for both conditions, so did some tweaking in parameters , Added red and green color threshols to detect lanes properly. 
 
# Failure points:
* Pipeline will fail when there is very dense forest or tunnel. 
* It will also fail in cases where huge white truck is crossing the same lane and white lane is partially visible. 
* It will fail when roads are in hilly region or there are bumpy roads. 

# To tackle above situations.
* Pipeline is required to split into 2 or more based on lighting conditions and paramters are required to tune to based on that. Single pipeline can not detect all things with same params.
* Take more samples of frames and try to match lane output of previous frame with current frame and see if there is huge anamoly or not. 
* Reduced size of nwindows when roads are on hilly region. Sample sliding window with small number of windows. 
