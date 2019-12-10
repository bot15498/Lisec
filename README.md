# Lisec: 3D object detection and segmentation using Lidar data


## Dependencies
The main dependencies are:
* Tensorflow (With Keras included)
* <a href="https://github.com/lyft/nuscenes-devkit">Lyft Dataset SDK</a>. 
* Shapely

All of these can be installed through pip3


## Data
The data used for training and testing is from the Lyft Autonomous Vehicle dataset. This data contains a series of 
samples of Lidar and camera data. This data can be downloaded from 
Kaggle <a href="https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data">here.</a> 
(Careful, it's 85 GB of data). 

For easier use, we have included some output models and predicted data so you don't need to download
the whole dataset to evaluate it.

We use this data in tandem with the Lyft-Object-Detection SDK. In order training and predictions to
work, you must specify the location of the Lyft Autonomous Vehicle dataset. This location is
saved in the Constants.py file.

## Training the Model
The model is very big, and will not fit in a normal amount of RAM. We recommend training
on a server or cluster with at least 500 GB of usable RAM.