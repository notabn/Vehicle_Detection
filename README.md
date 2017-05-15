**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video_output.mp4

---
###Writeup / README


###Histogram of Oriented Gradients (HOG)

####1. Extracted HOG features from the training images

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.feature.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.feature.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. HOG parameters.

I tried various combinations of color spaces and parameters of HOG. After trial and errors I came up with the combination of features extracted from  the color histogramm and the spatial color along with the HOG features from the color space `YCrCb`.  This  proved to give the best accuracy and object detection in the images.

####3. Training the classifier

The features decribed above were fed to a linear SVM classifier. For the training I used images from GTI and KITTI of car and non cars objects. 

###Sliding Window Search

I decided to search with a sliding window in the lower half plane of the image with a step of one cell at different scales all over the image and came up with :

![alt text][image3]


Ultimately I searched on up to three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. 

---

### Video Implementation

Here's a [link to my video result](./project_video.mp4)


#### False positives.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

####Here are six frames and their corresponding heatmaps:

![alt text][image5]

####Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

####Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

SVM classsifier prove to do a good job in detecting objects, considiring that also cars driving on the other side of the road were also idenfied. To deal with these false posives and also
other parts of the image that were also detected, I increased the threshhold in the heat map and also average the heatmao over 10 frames. However if a false
positive is detected over the 10 frames the average will fail.
To improve the robusness of the algorithm I would futher implement some sanity check on the size ratio of the bounding boxes.
The object detection fails when the objects are futher away from the ego vehicle and so goging in the higher half plane of the image. A reason is that the sliding widwod search was don only in the lowwer half plane.
Also a bigger dataset that includes samples with smaller objects and also a different classifier may improve the detection. 

Where the cars are two close to each other because of the heatmap they are detected as one object.

 

