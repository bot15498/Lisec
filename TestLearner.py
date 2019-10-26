#!/usr/bin/env python
# coding: utf-8

# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')
from lyft_dataset_sdk.lyftdataset import LyftDataset

# load dataset
level5Data = LyftDataset(
	data_path='E:\\CS539 Machine Learning\\3d-object-detection-for-autonomous-vehicles',
	json_path='E:\\CS539 Machine Learning\\3d-object-detection-for-autonomous-vehicles\\train_data',
	verbose=True
)


# In[33]:


# libraries
import numpy as np
import os
from pyquaternion import Quaternion
from matplotlib import pyplot as plt
from math import floor

# Set constants
dataDir = 'E:\\CS539 Machine Learning\\3d-object-detection-for-autonomous-vehicles'


# In[34]:


testScene = level5Data.scene[0]
testSample = level5Data.get('sample',testScene['first_sample_token'])


# In[35]:


# Test getting lidar points.
topLidarMetaData = level5Data.get('sample_data',testSample['data']['LIDAR_TOP'])
filePath = topLidarMetaData['filename'].replace('/','\\')
test = np.fromfile(os.path.join(dataDir, filePath), dtype=np.float32)
# reshape to get each point (5 values, then drop last 2 since they are intensiy (always 100?) and always 1?)
fixed = test.reshape(-1,5)[:,:3]
fixed


# In[36]:


level5Data.get('calibrated_sensor',topLidarMetaData['calibrated_sensor_token'])


# In[37]:


level5Data.render_sample(testSample['token'])


# In[38]:


# Uses quaternions to rotate all points in a scene to match the location of the lidar sensor on the car.
def rotate_points(points, rotation, inverse=False):
    quaternion = Quaternion(rotation)
    if inverse:
        quaternion = quaternion.inverse
    return np.dot(quaternion.rotation_matrix, points.T).T


# In[39]:


# Takes the sample dict and returns an array of n,3 with every point in the sample. 
def combine_lidar_data(sample):
    sensorTypes = ['LIDAR_TOP', 'LIDAR_FRONT_RIGHT', 'LIDAR_FRONT_LEFT']
    sensorFrameMetadata = [level5Data.get('sample_data',testSample['data'][x]) for x in sensorTypes]
    allPoints = []
    for sensorFrame in sensorFrameMetadata:
        sensor = level5Data.get('calibrated_sensor', sensorFrame['calibrated_sensor_token'])
        # get points
        filePath = sensorFrame['filename'].replace('/','\\')
        rawPoints = np.fromfile(os.path.join(dataDir, filePath), dtype=np.float32)
        
        # reshape to get each point (5 values, then drop last 2 since they are intensiy and always 1?)
        rawPoints = rawPoints.reshape(-1,5)[:,:3]
        
        # need to rotate points per sensor, then translate to position of sensor before combining
        points = rotate_points(rawPoints, sensor['rotation'])
        points = points + np.array(sensor['translation'])
        allPoints.append(points)
    allPoints = np.concatenate(allPoints)
        
    return allPoints
testSampleLidarPoints = combine_lidar_data(testSample)


# In[40]:


# Test view the resulting lidar data.
plt.figure(figsize=(12, 12))
plt.axis('equal')
plt.scatter(np.clip(testSampleLidarPoints[:,0],-50,50), np.clip(testSampleLidarPoints[:,1],-50,50), s=1, c='#000000')


# In[41]:


# given a x,y,z, find coordinate of voxel it woul be in. 
# Voxels are defined by their lower leftmost point
def get_voxel(point, xSize, ySize, zSize):
    x = floor(point[0] / xSize)
    y = floor(point[1] / ySize)
    z = floor(point[2] / zSize)
    return (x, y, z)


# In[55]:


# takes an array of size n,3 (every lidar point in sample) and returns array of points to pass into VFE
# Returns array of size n,6
def VFE_preprocessing(points, xSize, ySize, zSize, sampleSize, maxVoxelX, maxVoxelY, maxVoxelZ):
    clusteredPoints = {}
    # Iterate through points and add them to voxels
    for idx, point in enumerate(points):
        # expecting n to be around 200,000. Could be bad. Average time on local machine is about ~15 sec
        key = get_voxel(point, xSize, ySize, zSize)
        if -maxVoxelX < key[0] and key[0] < maxVoxelX                 and -maxVoxelY < key[1] and key[1] < maxVoxelY                 and -maxVoxelZ < key[2] and key[2] < maxVoxelZ:
            if key in clusteredPoints:
                clusteredPoints[key].append(idx)
            else:
                clusteredPoints[key] = [idx]
    # Add voxels that are empty
    for z in range(maxVoxelZ + 1):
        for y in range(-maxVoxelY, maxVoxelY + 1):
            for x in range(-maxVoxelX, maxVoxelX + 1):
                if (x, y, z) not in clusteredPoints:
                    clusteredPoints[(x, y, z)] = []
    # Sample points and fil the rest of the voxel if not full
    appendedPoints = []
    for voxel in clusteredPoints:
        # sample points, then find center
        s = sampleSize if len(clusteredPoints[voxel]) > sampleSize else len(clusteredPoints[voxel])
        if s > 0:
            sampleIdx = np.random.choice(clusteredPoints[voxel], size=s, replace=False)
            # get points for this voxel
            currPoints = points[sampleIdx]
            centroid = np.mean(currPoints, axis=0)
            # subtract constant x, y, z values of centroid from each column. Use 0:1 to keep it as a 2D array
            centroidX = currPoints[:,0:1] - centroid[0]
            centroidY = currPoints[:,1:2] - centroid[1]
            centroidZ = currPoints[:,2:3] - centroid[2]
            concat = np.hstack((currPoints, centroidX, centroidY, centroidZ))
            buffer = np.vstack((concat, np.zeros((sampleSize - s, 6))))
            appendedPoints.append(buffer)
        else:
            appendedPoints.append(np.zeros((sampleSize, 6)))
    return np.concatenate(appendedPoints)


# In[56]:


# size of voxel
voxelx = 0.4
voxely = 0.2
voxelz = 0.2

# Number of voxels in space that we care about
nx = int(100 / voxelx)
ny = int(100 / voxely)
nz = int(2 / voxelz)

# Limit of points per voxel to reduce size of data. 
maxPoints = 35


# In[57]:


import time
startTime = time.time()
testVFEPoints = VFE_preprocessing(testSampleLidarPoints, voxelx, voxely, voxelz, maxPoints, nx//2, ny//2, nz)
endTime = time.time()
print(endTime - startTime)
testVFEPoints


# In[54]:


# Keras time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'C:\\Users\\Skyler\\Documents\\Thats_Classified_Information\\Utilities\\graphviz\\bin'

#Input is a tensor that separates each voxel. Empty voxels are all 0. 
inLayer = Input(shape=(nx, ny, nz, maxPoints, 6), name='InputVoxel')
outLayer = addVFELayer(inLayer, 6, 16)
model = Model(inputs=inLayer, outputs=outLayer)
plot_model(model, show_shapes=True)


# In[17]:


def addVFELayer(layer, startNum, endNum):
    # FCN is linear, batch normalize, then relu
    layer = Dense(endNum)(layer)
    layer = BatchNormalization()(layer)
    layer = Dense(endNum, activation='relu')(layer)
    # now do the max pooling per 
    return layer





