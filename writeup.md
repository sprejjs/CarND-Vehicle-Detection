## Vehicle Tracking project by Vlad Spreys

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This file is my write up.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I have used two aproaches to extract the hog features and they are slightly different during the training and during the image search.

Durng the training I extract the HOG features from the whole image. The code is located in the `get_hog_features` method of my [Jupyter book](code.ipynb).

However, doing the same during the image processing would take too long as there are too many sliding windows and there would be too much processing power wasted. So, instead, I get the hog features from the whole image before I run the sliding windows. The code is available inside of my `find_cars` method of my notebook. Here is the relevant part:

```python
# Compute individual channel HOG features for the entire image
hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
```

####2. Explain how you settled on your final choice of HOG parameters.

I have tried various parameters during the training of my network and observed the change in accuracy. I ended up using the parameters which gave me the best accuracy despite the loss in computation speed.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Please refer to the "Training" section of my [Jupyter book](code.ipynb). I have used `SVC` classifier with `Linear` kernel and I have configured it to output predictions instead of classifications. Outputing predictions helped me to avoid false positive results as I only consider the output to be a vehicle when confidence of the classifier is higher 95%.

Here is the relevant code from my book for training:

```python
svc = SVC(kernel='linear' , probability = True)
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()

print('Using spatial binning of:',spatial,'and', histbin,'histogram bins')
print('Feature vector length:', len(X_train[0]))
print(round(t2-t, 2), 'Seconds to train SVC...')
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
```

And prediction:

```python
test_prediction = svc.predict_proba(test_features)
if test_prediction[0][1] > 0.95:
	#This is a car
```

Accuracy of my classifier on the test set is 98.96%

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code is available inside of the `find_cars` method of my [Jupyter book](code.ipynb). First, I have limited the area of interest to the road as there is no reason to search for the cars in the sky. Secondary, I have converted the entire image to `hog`, so that this operation is only performed once per image and run sliding windows inside of the area interest. 

I didn't need to change the parameters too dramatically as the default code given to us during the lectures provided very sophisticating results. 

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

Some example of my pipeline:

<img src="output_images/test1.jpg" align="left" height="150">
<img src="output_images/test2.jpg" align="left" height="150">
<img src="output_images/test3.jpg" align="left" height="150">
<img src="output_images/test4.jpg" align="left" height="150">
<img src="output_images/test5.jpg" align="left" height="150">
<img src="output_images/test6.jpg" align="left" height="150">


---
### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](output_videos/project_video.mp4).


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To avoid the false positive results I have changed the classifier to return the probabilities instead of classes and I only use the results which have over 95% confidence.

To combine multiple boxes together I have used a heat map of the bounding boxes and I have used the minimum and maximum values of the heatmaps to get the new bouding boxes. The code can be found in my [Jupyter book](code.ipynb) under the "Adding a HeatMap" section. Here is the relevant part:

```python
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
```


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Combining the training data from multiple sources was an interesting challenge. 

Aside from that, the biggest setback I had during the implementation of the project is when my classifier displayed a very high accuracy (over 98%), but I was getting too many false positive results during the testing of my pipeline. Turns out that this happened because the image pre-processing differed during the training and testing. In the future I will try to create a separate pre-processing function which I would re-use instead of copying and pasting the parameters. 

My pipeline is likely to fail during bad weather conditions including, but not limited to the fog, heavy rain and snow. I feel like computer vision is not advanced enough yet to deal with extreme weather.

Also, there are several false positives in the output of my video. I believe that the output result is good enough for the purpose of this project. In the real world scenario, though, we could track the position of a vehicle over several frames. The vehicles don't appear out of nowhere, so if one frame starts displaying a vehicle in the middle of a lane then it is likely a false positive. 
