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

The Lyft-Object-Detection SDK has a key object called the Level 5 Dataset.
This object reference is needed for most functions in this repo.

## Training the Model
The model is very big, and will not fit in a normal amount of RAM. We recommend training
on a server or cluster with at least 500 GB of usable RAM.

Training the model is done with the model_training.py. The two key functions
in this file are train() and train_with_model(). train() requires a list
of samples from the Level 5 Dataset to train on, the reference to the 
Level 5 Dataset, and a location to save the model to. train_with_model()
is a similar function but allows the user to load a model from disk.

## Predicting Using the Model
Predicting is done by running Predict.py. The main function in this file
is predictMain(), which requires a sample from the Level 5 Dataset, the 
Level 5 dataset object, the model to predict with, and an output path
to save the resulting numpy files.

## Visualizing Results
Converting predictions to bounding boxes is done with the rpnToRegion.py
script. The script loads the output of Predict.py as well as a reference
to the Level 5 Dataset. The program will then apply a transformation from
the RPN output to 20 potential cars in the sample. This result will be 
plotted with the original Lidar data and the actual annotations of the 
scene. The IoU comparison between the prediction with the ground truth 
will also be printed to the console. 

## Other Notes / Fixes
There are certain features of Keras and Tensorflow that prevent the network
from functioning smoothly. 

The first is Keras' inability to take 
SparseTensors as input to Keras layers. This has to do with the fact that
the shape of a SparseTensor is a Tensor, so you cannot represent an
input SparseTensor's variable size with a Tensor. The solution to this is 
to transform the input to the network to a dense Tensor. The consequence 
is the model does not train on most computer's memory. The model was trained
on a server with 512 GB of RAM and a AMD Opteron 6380 processor. 

Some Keras layers also do not have support for input Tensors of a rank
5 or higher. The major layer that caused this problem was the Dense 
layer. Most of our input Tensors are rank 6 Tensors in the VFE
layers. Our solution to this was to reshape the Tensor every time we 
wanted to pass the data through a Dense layer.



