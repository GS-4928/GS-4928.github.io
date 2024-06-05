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
      - [Utility Functions](#utils)
      - [Tracking Class](#tracking)
      - [Assigning Teams and Ball Possession](#ball-possession)
      - [Compensating for Camera Movement](#camera-movement)
      - [Transforming the Camera View](#view-transform)
      - [Speed and Distance Estimation](#speed-distance-estimator)
- [05. Pulling it All Together](#combination)

___

# Project Overview  <a name="overview-main"></a>

I wanted to learn more about object detection as a field of computer vision, seeing how it has evolved and what some of the current approaches are. As I'm a long suffering football fan, I thought that I would combine the two of these into a project!

We move through different data sets within this project, starting off with a kaggle data set of Bundesliga clips before settling on a labelled dataset from RoboFlow, an open source community for all things computer vision. Once our data is decided upon, we can utilise it to train up our YOLOv5 model, a model purpose built for real time object detection and the precision of its detections.

After the model has been trained, we can begin to write some python code to create classes and functions to successfully turn a blank video clip of a football match into a fully annotated, gameified visual experience!

<br>
___

# Data Overview  <a name="data-overview"></a>

There are lots of possible data sources that we could choose from. Initially, investigation began on a Kaggle data set : DFL - Bundesliga Data Shootout. This data had a large selection of 30 second clips from german first division football matches. The initial exploration of this data showed a potential problem - none of the potential objects are labelled within it! For basic purposes this is fine, but as we expand the capabilities of our scripting we will want to introduce some tracking features that require an associated tracking id to be present with each object.

To this end, we will utilise **RoboFlow** to choose and download a similar dataset for us to train our model on, the difference being that this new dataset will have each object of interest tagged for ease of tracking later on down the line. Below is a frame from the input video that we will be working on with our trained model.

<br>
![alt text](/img/posts/Input-video-frame.png "Initial Input Video Frame")
<br>

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

The above code demonstrates how simple it can be to get started with a YOLO model, utilising the ultralytics python library. Running our input video through this model shows that it is capable of identifying people and items within the frame, but that it doesn't provide some of the context that we would need for a football match.

<br>
![alt text](/img/posts/Model-without-tracking.png "Model output without tracking information")
<br>

The next step from here is to download the desired, labelled dataset from roboflow and pass it through to train our model. The training set expects our data to be in a specific form, sitting in a folder of the same name as its current folder otherwise it will crash when attempting to run! 

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
![alt text](/img/posts/Model-with-tracking.png "Model Output With Tracking Information Courtesy of Roboflow")
<br>
___

# Creating our Python Script <a name="script-creation"></a>

#### Utility Functions <a name="utils"></a>
To be able to both input and outpt video files from our model, we can employ the cv2 library to write a reading and saving video function.

```python
#looping over our input video to store our frames within a list
import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

#loop over our video frames and output the frames
def save_video(output_video_frames,output_video_path):
    #define an output format
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path,fourcc,24,
                          (output_video_frames[0].shape[1],output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
```
<br>
The video reading and saving functions are being saved in the utils folder within our folder structure. We're going to add another bunch of utility functions around objects within each video frame.

```python
def get_box_centre(bbox):
    #cast dimensions of bbox to coords
    x1,y1,x2,y2 = bbox
    #find centre point
    return int((x1+x2)/2),int((y1+y2)/2)

def get_box_width(bbox):
    #return x2 minus x1 for width
    return (bbox[2] - bbox[0])

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)
```
Creating an __init__.py file to expose our functions to the wider contents of our folders, including the main body of our function, main.py. We now have the logic in place to read our videos in and save them, what needs to be done is to add in the detection (keeping note of the bounding box/class of subjects in the video) and tracking (logic to tell us which bounding box belongs to which subject from frame to frame) aspects of the model. There are different ways of carrying this out and we will try to use a mix of visual features and predictions on the movement of each subject!

For ease of deployment within our main script, I decided to create a set of classes containing the requisite functions that will be needed to implement our desired annotations!

___
<br>

#### Tracking Class <a name="tracker"></a>

First up is a class to enable tracking of players, referees and the football itself.

```python

#create new tracker class
class Tracker:
    def __init__(self, model_path):
        #model set up on initiation
        self.model = YOLO(model_path)
        #tracker set up on instantiation
        self.tracker = sv.ByteTrack()

    #add position to tracks
    def add_position_to_tracks(self, tracks): ...

    #employ interpolation to fill in missing ball positions
    def interpolate_ball_position(self, ball_positions): ...
       
    #detect frames from our video files
    def detect_frames(self, frames): ...
        
    #function to draw an ellipse around the players
    def draw_ellipse(self, frame, bbox, colour, track_id=None): ...
    
    #function to draw a triangle above the ball
    def draw_triangle(self, frame, bbox, colour): ...
        
    #function to draw team possession stats
    def draw_team_possession(self, frame, frame_num, team_possession): ...
        
    #function to retrieve object tracks, either from existing stub or by running the code
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None): ...
       
    #annotate the frames, using the drawing functions defined above
    def draw_annotations(self,video_frames,tracks, team_possession): ...

```
For a full look at the code used within this class as well as the others employed, please check my github profile! Outside of the initialisation function which sets up an instance of our pre-trained model with an instance of a tracker courtesy of the pyhton library supervision, we have 8 methods that can be used by members of this class:
1. get_object_tracks - if this is being run for the first time, set up dictionaries for the players, referees and ball that we can then append tracking information to. If these tracks already exist, then we can load them before exiting the function
2. add_position_to_tracks - passing through the obtained tracking information and linking it up with an associated bounding box for each object within frame
3. interpolate_ball_position - for when the bounding box of the ball isn't found within the tracking information, we can carry out interpolation to fill in the gaps
4. detect_frames - we want to pass through video data and then apply our pre-trained model to provide tracking information and predictions of objects within the frame
5. draw_ellipse - around the players within the image we want to draw an allipse beneath their feet that follows them as they move. If a track_id is given then we also want to add this to our annotation as well (this will be used for our players, something different will be employed for the ball and the referees will be ignored!)
6. draw_triangle - speaking of tracking the ball, this is what the triangle will be used for! Appearing as a small triangle above the ball and taking advantage of the interpolated ball position to more closely follow its movement
7. draw_team_possession - counting the number of frames that each team has possession of the ball for, followed with an annotation to show this changing statistic across the duration of the clip. We will explore how to calculate team possession through another Class
8. draw_annotations - pull all of this information together into one cohesive annotation applied to each frame, utilising the previous functions created

We've covered teh tracking information that we need to pull out from this, now we need to cover which team has the ball to ensure that our annotaitons are correct!
___
<br>

#### Assigning Teams and Ball Possession <a name="camera-movement"></a>

Here's where we rely on a trusty machine learning technique: clustering! To determine which team has possession of the ball at any one time, we can take the bounding box for each player and perform k-means clustering to determine the shirt colour of within that bounding box. As all members of the same team, apart from the goalkeeper, are wearing the same colour, we can determine which player belongs to which team and then calculate our desired statistics from there, as well as inform our tracking data more fully!

```python
#create class to assign players to teams
class TeamAssigner:

    def __init__(self):
        #set up dictionary of team colours
        self.team_colours = {}
        #set up dictionary of which player belongs to which team
        self.player_team_dict = {}

    #function to produce our clustering model for each image
    def get_clustering_model(self, image):...

    #function deciding on the colour of each player
    def get_player_colour(self, frame, bbox):...

    #function to assign team colours based on player colours
    def assign_team_colour(self, frame, player_detections):...

    #function to assign players to each team
    def get_player_team(self, frame, player_bbox, player_id):...
    
```
A smaller class than the tracker class previously created, this will allow us to produce a clustering model for each player within a video, deduce the colour for each player and assign colours for each team based on these sets of colours. Finally, we want to then assign each player to a respective team. For the goalkeeper of the white team specifically, I decided to hardcode his team id to match the wider team as the kmeans model had a difficult time differentiating between the two teams for the goalkeeper.

___
<br>

#### Compensating for Camera Movement <a name="ball-possession"></a>

Whilst we have a good chunk of the information we need to provide tracking details and possession details on the football match as it goes on, what we don't have a way of doing yet is estimating the distances and speeds at which the players are moving. To begin this process, we need to account for the position and movement of the camera taking the video. The reason for this is that we want to adjust the positions of the players within the frame to take account of how the camera moves, ensuring that the tracking information we have is more accurate. This will involve estimating the camera movement at any one time and subtracting it from a players' current position that is already stored within one of our tracking dictionaries.

```python

class CameraMovementEstimator:
    
    def __init__(self,frame):

        #minimum camera movement
        self.minimum_distance = 5
        #specify lk params
        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, 10, 0.03)
        )

        first_frame_greyscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #choosing features from the top and bottom of the frame that shouldn't be obstructed by movement
        mask_features = np.zeros_like(first_frame_greyscale)
        #slicing out so we have the top 20 rows of pixels and the bottom banners
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1

        #setting up dictionary to use for feature extraction
        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features
        )

    #adjust object positions to account for camera movement
    def adjust_positions_to_tracks(self,tracks,camera_movement_per_frame):...

    #determine the camera movement
    def get_camera_movement(self,frames,read_from_Stub=False,stub_path=None):...

    #annotate each frame with this live camera movement
    def draw_camera_movement(self,frames,camera_movement_per_frame):...
```
The initialisation of this class selects a range of pixels across the image that should be unobstructed by movement within the running of the clip, allowing for comparative calculation of camera movement. It also sets parameters for use within cv2 functions that will help with calculating the positional movement of the camera and the subsequent adjustments that need to be made.

___
<br>
#### Transforming the Camera View <a name="view-transform"></a>

The perspective of our camera works well to capture a larger area of the football pitch for watchers at home, but raises issues for us if we want to accurately estimate the speed and distance that objects are travelling at within frame. Items further back in the frame are shortened and narrowed due to the perspective whilst items closer to the front of the frame appear larger and wider. This perspective change needs to be compensated for.

```python

class ViewTransformer():

    def __init__(self):
        #width of football pitch
        pitch_w = 68
        #length of segment of football pitch
        pitch_l = 23.32

        #provide the position of the points that map with this section of the pitch
        self.pixel_verticies = np.array([
            [110,1035],
            [265,275],
            [910,260],
            [1640,915]
        ])

        #provide the target corners of the size of the pitch that the trapezoid corresponds to
        self.target_verticies = np.array([
            [0,pitch_w],
            [0,0],
            [pitch_l,0],
            [pitch_l,pitch_w]
        ])

        #cast each set of corners as floats
        self.pixel_verticies = self.pixel_verticies.astype(np.float32)
        self.target_verticies = self.target_verticies.astype(np.float32)

        #transform the image into the desired measurements
        self.perpective_transformer = cv2.getPerspectiveTransform(self.pixel_verticies,self.target_verticies)

    #transform the points based on transformer
    def transform_point(self,point):...

    #add this transformed position to the tracks dictionary
    def add_transformed_position_to_tracks(self, tracks):...

```

Within the initilisation of this class, we preset some variables to define the real length and width of a section of the pitch, and then map them to the corresponding pixels within the video. We do this as we want to effectively transform the shape of the uneven trapezoid that represents our pitch section into a match for the real dimensions of that rectangular area within our video frame. To this end we employ a perspective transformation thanks to cv2 that we can call later throughout the class methods. There are only two methods contained within here; one to transform points based on the perspective transformer and another to leverage this point transformer to adjust the positional tracking information for each object within the tracking dictionary we have already set up.
___
<br>
#### Speed and Distance Estimation <a name="speed-distance-estimator"></a>

Nearly there! One of the final steps to do before tying all our classes and utility functions together is the estimation of speed and distance travelled by eaech player throughout the duration of our input video. With our cmaera movement estimator set up, we can compensate for the skewed perspective of our input video to estimate distances and speeds more accurately. The only bit of hard coding within the upcoming class will be the frame rate of our input video and the number of frames that we want to be measuring our speed and distance changes over.

```python

class SpeedAndDistanceEstimator():

    def __init__(self):
        #define window to measure speed over
        self.frame_window = 5
        self.frame_rate = 24

  #calculate the speed and distance travelled from previous positions within each frame and add these to the tracking dictionary
  def add_speed_and_distance_to_tracks(self,tracks):...

  #annotate the frame with these stats
  def draw_speed_distance_annotations(self,video_frames,tracks):...

```
With this class we have everything we need to take in raw video files and, through the application of our pre-trained YOLO model, output a tracked, annotated video file showing statistical information about the flow of the football game!

___
<br>
# Pulling it All Together <a name="combination"></a>

We have the tools now to convert an input video into a fully annotated out put video! Before anything, we need to import all of our classes and other libraries that will be needed for this to run successfully.

```python

import cv2
from utils import read_video, save_video, measure_distance
from trackers import Tracker
from team_assigner import TeamAssigner
from ball_possession import BallPossession
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator
```
With these classes and libraries available, we can read in our video from its file path, and begin the process of tracking objects within the frames

```python
def main():
    #read in video from our input video folder
    video_frames = read_video(input_file_path)

    #initialise tracker
    tracker = Tracker(model_file_path)

    #track our video frames
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=stub_file_path)
    
    #calculate object positions
    tracker.add_position_to_tracks(tracks)
```
Our tracking dictionaries will be fully populated now for each frame of our input video. Adjustments due to the movement of the camera cna be made next

```python
    #track the camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path =camera_stub_path)
    
    #adjust positions to account for camera movement
    camera_movement_estimator.adjust_positions_to_tracks(tracks, camera_movement_per_frame)
```
We want to ensure that the camera view is transformed to take account of the perspective issues, and then update the tracking information for the ball to include the interpolated information that will be calculated for missing frames

```python
    #transform view to reflect true dimensions of pitch
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    #interpolate the position of the ball for each missing frame
    tracks['ball'] = tracker.interpolate_ball_position(tracks['ball'])
```
Setting up the speed and distance estimator class comes next, followed by an instance of the team assigning class to allow us to prepare to calculate the possession of the ball. With this team assigner, we can loop through each frame to allocate our players to the appropriate teams

```python
    #estimate speed and distance travelled
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    #assign players to teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_colour(video_frames[0],
                                     tracks['players'][0])
    
    #for each player, assign a team and appropriate team colour
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_colour'] = team_assigner.team_colours[team]
```
Next comes our posession figures, utilising the ball possession class we created earlier.

```python
    #assign ball possession
    ball_possession = BallPossession()
    team_possession = []

    for frame_num, player_track in enumerate(tracks['players']):
        #pull out ball bounding box
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        #see which player is closest to the ball
        assigned_player = ball_possession.assign_ball_possession(player_track, ball_bbox)

        #if the assigned player value has changed, update the possession of the ball
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_possession.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_possession.append(len(team_possession))

    #cast the possession to a numpy array for use within the annotations
    team_possession = np.array(team_possession)
```
The final line above transformed the possession figures into an array, so that we can pass it through our drawing functions without raising an error. This is the final step: drawing our tracking information, the camera movement and the statistical information about the players and teams. The only thing to do after that is to save our output video frames.

```python
    #draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames,tracks, team_possession)

    #draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,
                                                                        camera_movement_per_frame)
    
    #draw speed and distance
    speed_and_distance_estimator.draw_speed_distance_annotations(output_video_frames,tracks)
    #save our video
    save_video(output_video_frames, 'Output_Videos/08fd33_4_output_final.avi')
```

And that does it! Running this code will take in our input video and annotate it to produce this!

<br>
![alt text](/img/posts/output-video-frame.png "Final Video Output Frame")
<br>
___
<br>
# Discussion, Growth & Next Steps <a name="growth-next-steps"></a>

If this were to be put into production as a scouting tool for use by clubs, there could be further development scope to create a web applcation through something like Streamlit or Shiny to allow for the input of raw video clips that then produce annotated output clips with information on possession, speed and distance travelled.

Depending on stakeholder feedback, future functionality surrounding number of passes, direction of pass, length of pass and other analytical metrics could be included to create a more holistic output with a wider ranging use for a recruitment department. As an proof of concept though, it works well! A more finely tuned model explicitly trained on football matches may provide a slight performance improvement, if only for the ball tracking an occassional goalkeeper identification issue, but the YOLOv5 model performed as needed for this project, thanks to the benefits of transfer learning within CNNs.

I think that this type of model is widely applicable to most pitch based sports that are televised/videoed, with some tweaks to the initial training of the pre-defined model depending on the data (hockey, rugby, lacrosse etc.)

Overall, I'm proud of this work, and I hope it's been interesting for oyu to read along! For a full breakdown of the code used, visit my github page.
