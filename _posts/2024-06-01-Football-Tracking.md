---
layout: post
title: In Progress - Utilising Computer Vision to Track Player Movement in a Football Match
image: "/posts/football-pitch-title.jpg"
tags: [CV2, YOLO, Data Science, Computer Vision, Python, OOP]
---

After completing projects based on Neural Networks and how they can be leveraged for image classification, I wanted to work on further Computer Vision tasks! This project is the result of that desire; leveraging open source python libraries to track and produce stats on players in a clip from a football match. Utilising Object Detection will allow for the location of instances of specific objects within an image or video, in this case football players, referees and the football itself.

# Table of contents

- [00. Project Overview](#overview-main)
- [01. Data Overview](#data-overview)
- [02. Object Detection Overview](#object-detection-overview)
- [03. Setting Up YOLO Instance](#YOLO-setup)
- [04. Creating our Python Script](#script-creation)
- [05. Execute Search](#execute-search)
- [06. Discussion, Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

I wanted to learn more about object detection as a field of computer vision, seeing how it has evolved and what some of the current approaches are. As I'm a long suffering football fan, I thought that I would combine the two of these into a project!

We move through different data sets within this project, starting off with a kaggle data set of Bundesliga clips before settling on a labelled dataset from RoboFlow, an open source community for all things computer vision. Once our data is decided upon, we can utilise it to train up our YOLOv5 model, a model purpose built for real time object detection and the precision of its detections.

After the model has been trained, we can begin to write some python code to create classes and functions to successfully turn a blank video clip of a football match into a fully annotated, gameified visual experience!

<br>
___

# Data Overview  <a name="data-overview"></a>

There are lots of possible data sources that we could choose from. Initially, investigation began on a Kaggle data set : DFL - Bundesliga Data Shootout. This data had a large selection of 30 second clips from german first division football matches. The initial exploration of this data showed a potential problem - none of the potential objects are labelled within it! For basic purposes this is fine, but as we expand the capabilities of our scripting we will want to introduce some tracking features that require an associated tracking id to be present with each object.

To this end, we will utilise **RoboFlow** to choose and download a similar dataset for us to train our model on, the difference being that this new dataset will have each object of interest tagged for ease of tracking later on down the line.

Enough about the data for now, we need to explore what exactly Object Detection is!
<br>
___

# Object Detection Overview  <a name="object-detection-overview"></a>

<br>
#### Overview

Object Detection is a very popular task within the realm of computer vision. In its simplest form, Object Detection seeks to analyse images or videos and locate instances of specific objects within these images/videos. For example, footage from a camera set up within a nature reserve to monitor activity within the space could be analysed with object detection to see the different kinds of animals coming and going from the area! MOre importantly for the purpose of this project, object detection isn't limited to identifying one object at a time and in fact multiple methods have been developed to enable the identification of multiple objects within the same frame.

<br>
![alt text](/img/posts/example-detection.png "An Example of Object Detection")
<br>

In the above image we can see object detection at work in a more rudimentary fashion, being capable of identifying animals within an image even if it can't directly tell us that these are lions. Traditionally there were three steps to Object Detection:
1. Target Region Selection - The process of bounding objects within an image. Originally this was done through trial and error for each class until an average bounding size could be achieved for that class.
2. Feature Extraction - With the objects bounded, the make up of each class was then analysed to pull out features that would describe each class, through tools like Scale-Invariant Feature Transform (SIFT) or Histogram of Oriented Gradients (HOG), although these tools could struggle with noise, illumination and image scale when extracting features.
3. Classification/Regression - With features now extracted for each class from the training set, classification would be carried out on the bounded objects within the images to predict which class each belonged to, through the use of Random Forest models or Support Vector Machines. Regression is also employed to ascertain the specific locations of each bounding box for each object.

This traditional approach suffered from a multitude of problems, including being computationally expensive, difficult to fine tune and being prone to overfitting. This changed with the advent of Deep Learning.

<br>
#### Current Types of Object Detection

Deep Learning, specifically the arrival of Convolutional Neural Networks and its progeny, has served to fill in some of the gaps around generalisation and computational expense. There are two main methods of object detection available: Two-Stage Detectors and One-Shot Detectors.

**Two-Stage Detectors** work through Region Proposal Networks. After an initial pass of the entire image through a lightweight CNN to generate feature maps, Regions of Interest (ROIs) are established through these feature maps fed in via the convolutional layers preceeding the pooling layer instead of focusing on a granular pixel-level approach, allowing for a much faster generation of ROIs. This was first seen within Faster Recursive Convolutional Neural Networks (Faster RCNN). Once these ROIs are produced, regression and classification are carried out to find the bounding box coordinates and the predicted class respectively. This consitutes the two step process.

**One-Stage Detectors** bypass Region Proposal Networks and Regions of Interest. Instead, the network predicts the fixed amount of probabilities at a given time from an image and directly performs regression/classification to map a bounding box and class probability onto image pixels. This process is incredibly fast but does have the potential to lose accuracy.

<br>
![alt text](/img/posts/Two-stage-vs-one-stage-object-detection-models-3179193683.png "Two-Stage vs One-Stage Detection: Architecture")
<br>

For this project, we will be utilising a One-Shot Detector model called YOLO, a multi-object detection algorithm.

___

# Setting Up YOLO Instance  <a name="YOLO-setup"></a>

YOLO - an acronym for You Only Look Once - was first introduced in 2015 by the team of  Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. It is all based around Precision, specifically the mean average precision. Precision in statistical terms is the ratio of true positive predictions to the total number of positive data points within a dataset.

Architecturally, YOLO has 3 distinct layers to it: 
1. A CNN **backbone** to produce assorted feature maps from the visual data.
2. A combined set of pooling layers that integrate and blend the different feature maps together to achieve a more generalised set of features, called a **neck**.
3. The **Head**, a dense output layer that takes in the generalised features alongside predictions for the bounding boxes within the visual sample. Classification of the class within the bounding box is carried out alongside regression for the features and bounding box position, resulting in a transformed image or video effectively annotated with class information.

YOLO outputs four pieces of information in two possible formats. The first format is the x and y coordinates of the subject within an image and the width and height of the bounding box that surrounds that subject. The second format is known as x, y, x, y and outputs the x and y coordinates of the top left of the bounding box along witht he coordinates for the bottom right of the same bounding box.

<br>

```python
#import required libraries
from ultralytics import YOLO

#define file path for input video
input_video = '...'

#instantiate model, can choose from small, medium, large, extra large and nano
model = YOLO('yolov8x')

#save our results to an object for ease of access and analysis
results = model.predict(input_video, save=True)

```

<br>

The above code demonstrates how simple it can be to get started with a YOLO model, utilising the ultralytics python library. The next step from here is to download the desired, labelled dataset from roboflow and pass it through to train our model. The training set expects our data to be in a specific form, sitting in a folder of the same name as its current folder otherwise it will crash when attempting to run!

```python

# import the required python libraries
from roboflow import RoboFlow
import shutil

# Set up our data to be passed into the YOLO model
rf = Roboflow(api_key=...)
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(1)
dataset = version.download("yolov5")

#enusre that directory is set up to be read correctly by model
data_forms = ['train','test','valid']
folder_location = dataset.location
add_on = 'football-players-detection-1'

#move train, test, and validation folders from one location to another utilising shutil
for i,j in enumerate(data_forms):
  shutil.move(folder_location+'/'+data_forms[i],folder_location+'/'+add_on+'/'+data_forms[i])

```

<br>
Our data comes with a .yaml file that provides important information on the class breakdown of the training, testing and validation datasets in use. With our data in the expected place now, we can begin to train our model.

```python
#train our model, with various inputs to tweak as desired
!yolo task=detect mode=train model=yolov5lu.pt data={dataset.location}/data.yaml epochs=100 imgsz=416 cache
```
The above command can be broken down into its constituent parts:
1. task = detect, we are setting up our model to do some detection work
2. mode = train, this incoming data will be used to train our model
3. model = yolov5l.pt, the model is selected (the large version of YOLOv5)
4. data = {dataset.location}/data.yaml, we specify where our data is as well as the associated configuration file to allow for classes and bounding information to be incorporated
5. epoch = 100, we decide how many iterations through the data we want our model to make
6. imgsz = 640, we can set the size of our input images

The output of this training, besides annotated images, is a pair of models labelled 'best' and 'last' that contain the best performing run nad the last performed run respectively. If we feed our training video into this best model of ours, we can immediately see some differences in performance and identification!
<br>
___

# Creating our Python Script <a name="script-creation"></a>

To be able to both input and outpt video files for training through our 

```python

# image pre-processing function
def preprocess_image(filepath):
    
    image = load_img(filepath, target_size = (img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = preprocess_input(image)
    
    return image

# image featurisation function
def featurise_image(image):
    
    feature_vector = model.predict(image)
    
    return feature_vector

```
<br>
The *preprocess_image* function does the following:

* Receives the filepath of an image
* Loads the image in
* Turns the image into an array
* Adds in the "batch" dimension for the array that Keras is expecting
* Applies the custom pre-processing logic for VGG16 that we imported from Keras
* Returns the image as an array

The *featurise_image* function does the following:

* Receives the image as an array
* Passes the array through the VGG16 architecture
* Returns the feature vector

<br>
#### Setup

In the code below, we:

* Specify the directory of the base-set of images
* Set up empty list to append our image filenames (for future lookup)
* Set up empty array to append our base-set feature vectors

```python

# source directory for base images
source_dir = 'data/'

# empty objects to append to
filename_store = []
feature_vector_store = np.empty((0,512))

```

<br>
#### Preprocess & Featurise Base-Set Images

We now want to preprocess & feature all 300 images in our base-set.  To do this we execute a loop and apply the two functions we created earlier.  For each image, we append the filename, and the feature vector to stores.  We then save these stores, for future use when a search is executed.

```python

# pass in & featurise base image set
for image in listdir(source_dir):
    
    print(image)
    
    # append image filename for future lookup
    filename_store.append(source_dir + image)
    
    # preprocess the image
    preprocessed_image = preprocess_image(source_dir + image)
    
    # extract the feature vector
    feature_vector = featurise_image(preprocessed_image)
    
    # append feature vector for similarity calculations
    feature_vector_store = np.append(feature_vector_store, feature_vector, axis = 0)

# save key objects for future use
pickle.dump(filename_store, open('models/filename_store.p', 'wb'))
pickle.dump(feature_vector_store, open('models/feature_vector_store.p', 'wb'))

```

___
<br>
# Execute Search <a name="execute-search"></a>

With the base-set featurised, we can now run a search on a new image from a customer!

<br>
#### Setup

In the code below, we:

* Load in our VGG16 model
* Load in our filename store & feature vector store
* Specify the search image file
* Specify the number of search results we want

```python

# load in required objects
model = load_model('models/vgg16_search_engine.h5', compile = False)
filename_store = pickle.load(open('models/filename_store.p', 'rb'))
feature_vector_store = pickle.load(open('models/feature_vector_store.p', 'rb'))

# search parameters
search_results_n = 8
search_image = 'search_image_02.jpg'

```
<br>
The search image we are going to use for illustration here is below:

<br>
![alt text](/img/posts/search-engine-search1.jpg "VGG16 Architecture")

<br>
#### Preprocess & Featurise Search Image

Using the same helper functions, we apply the preprocessing & featurising logic to the search image - the output again being a vector containing 512 numeric values.

```python

# preprocess & featurise search image
preprocessed_image = preprocess_image(search_image)
search_feature_vector = featurise_image(preprocessed_image)

```

<br>
#### Locate Most Similar Images Using Cosine Similarity

At this point, we have our search image existing as a 512 length feature vector, and we need to compare that feature vector to the feature vectors of all our base images.

When that is done, we need to understand which of those base image feature vectors are most like the feature vector of our search image, and more specifically, we need to return the eight most closely matched, as that is what we specified above.

To do this, we use the *NearestNeighbors* class from *scikit-learn* and we will apply the *Cosine Distance* metric to calculate the angle of difference between the feature vectors.

**Cosine Distance** essentially measures the angle between any two vectors, and it looks to see whether the two vectors are pointing in a similar direction or not.  The more similar the direction the vectors are pointing, the smaller the angle between them in space and the more different the direction the LARGER the angle between them in space. This angle gives us our cosine distance score.

By calculating this score between our search image vector and each of our base image vectors, we can be returned the images with the eight lowest cosine scores - and these will be our eight most similar images, at least in terms of the feature vector representation that comes from our VGG16 network!

In the code below, we:

* Instantiate the Nearest Neighbours logic and specify our metric as Cosine Similarity
* Apply this to our *feature_vector_store* object (that contains a 512 length feature vector for each of our 300 base-set images)
* Pass in our *search_feature_vector* object into the fitted Nearest Neighbors object.  This will find the eight nearest base feature vectors, and for each it will return (a) the cosine distance, and (b) the index of that feature vector from our *feature_vector_store* object.
* Convert the outputs from arrays to lists (for ease when plotting the results)
* Create a list of filenames for the eight most similar base-set images

```python

# instantiate nearest neighbours logic
image_neighbours = NearestNeighbors(n_neighbors = search_results_n, metric = 'cosine')

# apply to our feature vector store
image_neighbours.fit(feature_vector_store)

# return search results for search image (distances & indices)
image_distances, image_indices = image_neighbours.kneighbors(search_feature_vector)

# convert closest image indices & distances to lists
image_indices = list(image_indices[0])
image_distances = list(image_distances[0])

# get list of filenames for search results
search_result_files = [filename_store[i] for i in image_indices]

```

<br>
#### Plot Search Results

We now have all of the information about the eight most similar images to our search image - let's see how well it worked by plotting those images!

We plot them in order from most similar to least similar, and include the cosine distance score for reference (smaller is closer, or more similar)

```python

# plot search results
plt.figure(figsize=(20,15))
for counter, result_file in enumerate(search_result_files):    
    image = load_img(result_file)
    ax = plt.subplot(3, 3, counter+1)
    plt.imshow(image)
    plt.text(0, -5, round(image_distances[counter],3), fontsize=28)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

```
<br>
The search image, and search results are below:

**Search Image**
<br>
![alt text](/img/posts/search-engine-search1.jpg "Search 1: Search Image")
<br>
<br>
**Search Results**
![alt text](/img/posts/search-engine-search1-results.png "Search 1: Search Results")

<br>
Very impressive results!  From the 300 base-set images, these are the eight that have been deemed to be *most similar*!

<br>
Let's take a look at a second search image...

**Search Image**
<br>
![alt text](/img/posts/search-engine-search2.jpg "Search 2: Search Image")
<br>
<br>
**Search Results**
![alt text](/img/posts/search-engine-search2-results.png "Search 2: Search Results")

<br>
Again, these have come out really well - the features from VGG16 combined with Cosine Similarity have done a great job!

___
<br>
# Discussion, Growth & Next Steps <a name="growth-next-steps"></a>

The way we have coded this up is very much for the "proof of concept".  In practice we would definitely have the last section of the code (where we submit a search) isolated, and running from all of the saved objects that we need - we wouldn't include it in a single script like we have here.

Also, rather than having to fit the Nearest Neighbours to our *feature_vector_store* each time a search is submitted, we could store that object as well.

When applying this in production, we also may want to code up a script that easily adds or removes images from the feature store.  The products that are available in the clients store would be changing all the time, so we'd want a nice easy way to add new feature vectors to the feature_vector_store object - and also potentially a way to remove search results coming back if that product was out of stock, or no longer part of the suite of products that were sold.

Most likely, in production, this would just return a list of filepaths that the client's website could then pull forward as required - the matplotlib code is just for us to see it in action manually!

This was tested only in one category, we would want to test on a broader array of categories - most likely having a saved network for each to avoid irrelevant predictions.

We only looked at Cosine Similarity here, it would be interesting to investigate other distance metrics.

It would be beneficial to come up with a way to quantify the quality of the search results.  This could come from customer feedback, or from click-through rates on the site.

Here we utilised VGG16. It would be worthwhile testing other available pre-trained networks such as ResNet, Inception, and the DenseNet networks.
