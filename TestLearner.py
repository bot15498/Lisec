import matplotlib
matplotlib.use('agg')
from lyft_dataset_sdk.lyftdataset import LyftDataset
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Layer, Concatenate, Conv3D, ZeroPadding3D, \
	Reshape, Permute, ZeroPadding2D, Conv2D, Conv2DTranspose, Activation
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as tf_backend
import tensorflow as tf
from tensorflow.keras import optimizers

import numpy as np
from pyquaternion import Quaternion
from matplotlib import pyplot as plt
from math import floor
import os
from tensorflow import SparseTensor, sparse
import math
from shapely.geometry import Polygon

from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()
print(tf.executing_eagerly())

# constants
# size of voxel
voxelx = 0.5
voxely = 0.25
voxelz = 0.25

# max size of space to look at


# Number of voxels in space that we care about
nx = int(100 / voxelx)  # -50 to 50 m
ny = int(100 / voxely)
nz = int(2 / voxelz)

# number of anchors
anchors = [[1.6, 3.9, 1.56], [3.9, 1.6, 1.56]]
iouLowerBound = 0.45
iouUpperBound = 0.6

# Limit of points per voxel to reduce size of data.
maxPoints = 35

# index of points in input tensor
pointIndex = -2

# map of categories:
catToNum = {
	'car': 0,
	'pedestrian': 1,
	'animal': 2,
	'other_vehicle': 3,
	'bus': 4,
	'motorcycle': 5,
	'truck': 6,
	'emergency_vehicle': 7,
	'bicycle': 8
}

# # load dataset
level5Data = LyftDataset(
	data_path='C:\\Users\\snkim\\Desktop\\poject\\data',
	json_path='C:\\Users\\snkim\\Desktop\\poject\\data\\train_data',
	verbose=True
)


# helper layer that transforms the (None, 250, 500, 10, 1, 6) into (None, 250, 500, 10, 35, 6) for concat
class RepeatLayer(Layer):
	def __init__(self, **kwargs):
		super(RepeatLayer, self).__init__(**kwargs)

	def compute_output_shape(self, inputShape):
		return inputShape[:pointIndex] + (maxPoints,) + inputShape[pointIndex + 1:]

	def call(self, inputs, **kwargs):
		return tf_backend.repeat_elements(inputs, maxPoints, pointIndex)


# special pooling layer for VFE block
class MaxPoolingVFELayer(Layer):
	def __init__(self, combine=False, **kwargs):
		super(MaxPoolingVFELayer, self).__init__(**kwargs)
		self.combineDim = combine

	def compute_output_shape(self, inputShape):
		if not self.combineDim:
			return inputShape[:pointIndex] + (1,) + inputShape[pointIndex + 1:]
		else:
			return inputShape[:pointIndex] + inputShape[pointIndex + 1:]

	def call(self, inputs, **kwargs):
		return tf_backend.max(inputs, axis=pointIndex, keepdims=not self.combineDim)

	def get_config(self):
		baseConfig = super(MaxPoolingVFELayer, self).get_config()
		baseConfig['combine'] = self.combineDim
		return baseConfig


# Uses quaternions to rotate all points in a scene to match the location of the lidar sensor on the car.
def rotate_points(points, rotation, inverse=False):
	quaternion = Quaternion(rotation)
	if inverse:
		quaternion = quaternion.inverse
	return np.dot(quaternion.rotation_matrix, points.T).T


# Takes the sample dict and returns an array of n,3 with every point in the sample.
def combine_lidar_data(sample, dataDir):
	sensorTypes = ['LIDAR_TOP', 'LIDAR_FRONT_RIGHT', 'LIDAR_FRONT_LEFT']
	# Account for not all samples having all liar data for some reason
	actualSensorTypes = []
	for sensorType in sensorTypes:
		if sensorType in sample['data']:
			actualSensorTypes.append(sensorType)

	sensorFrameMetadata = [level5Data.get('sample_data', sample['data'][x]) for x in actualSensorTypes]
	allPoints = []
	for sensorFrame in sensorFrameMetadata:
		sensor = level5Data.get('calibrated_sensor', sensorFrame['calibrated_sensor_token'])
		# get points
		filePath = sensorFrame['filename'].replace('/', '\\')
		rawPoints = np.fromfile(os.path.join(dataDir, filePath), dtype=np.float32)

		# reshape to get each point (5 values, then drop last 2 since they are intensiy and always 1?)
		rawPoints = rawPoints.reshape(-1, 5)[:, :3]

		# need to rotate points per sensor, then translate to position of sensor before combining
		points = rotate_points(rawPoints, sensor['rotation'])
		points = points + np.array(sensor['translation'])
		allPoints.append(points)
	allPoints = np.concatenate(allPoints)

	return allPoints


# given a x,y,z, find coordinate of voxel it woul be in.
# Voxels are defined by their lower leftmost point
def get_voxel(point, xSize, ySize, zSize):
	x = floor(point[0] / xSize)
	y = floor(point[1] / ySize)
	z = floor(point[2] / zSize)
	return (x, y, z)


# takes an array of size n,3 (every lidar point in sample) and returns array of points to pass into VFE
# Returns array of size n,6
def VFE_preprocessing(points, xSize, ySize, zSize, sampleSize, maxVoxelX, maxVoxelY, maxVoxelZ):
	clusteredPoints = {}
	# Iterate through points and add them to voxels
	for idx, point in enumerate(points):
		# expecting n to be around 200,000. Could be bad. Average time on local machine is about ~15 sec
		key = get_voxel(point, xSize, ySize, zSize)
		if -maxVoxelX < key[0] and key[0] < maxVoxelX \
				and -maxVoxelY < key[1] and key[1] < maxVoxelY \
				and 0 < key[2] and key[2] < maxVoxelZ:
			# remove negatives.
			fixedKey = (key[0] + maxVoxelX, key[1] + maxVoxelY, key[2])
			if fixedKey in clusteredPoints:
				clusteredPoints[fixedKey].append(idx)
			else:
				clusteredPoints[fixedKey] = [idx]
	# Sample points and fil the rest of the voxel if not full
	appendedPoints = {}
	for voxel in clusteredPoints:
		# sample points, then find center
		s = sampleSize if len(clusteredPoints[voxel]) > sampleSize else len(clusteredPoints[voxel])
		sampleIdx = np.random.choice(clusteredPoints[voxel], size=s, replace=False)
		# get points for this voxel
		currPoints = points[sampleIdx]
		centroid = np.mean(currPoints, axis=0)
		# subtract constant x, y, z values of centroid from each column. Use 0:1 to keep it as a 2D array
		centroidX = currPoints[:, 0:1] - centroid[0]
		centroidY = currPoints[:, 1:2] - centroid[1]
		centroidZ = currPoints[:, 2:3] - centroid[2]
		concat = np.hstack((currPoints, centroidX, centroidY, centroidZ))
		buffer = np.vstack((concat, np.zeros((sampleSize - s, 6))))
		appendedPoints[voxel] = buffer
	indices = []
	values = []
	for voxel in appendedPoints:
		for i in range(len(appendedPoints[voxel])):
			for j in range(len(appendedPoints[voxel][i])):
				indices.append((voxel[2],) + voxel[:2] + (i, j))
				values.append(appendedPoints[voxel][i][j])
	# return as z, x, y
	return SparseTensor(indices=indices, values=values,
						dense_shape=[maxVoxelZ, maxVoxelX * 2, maxVoxelY * 2, sampleSize, 6])


def addVFELayer(layer, startNum, endNum):
	# FCN is linear, batch normalize, then relu
	actualEndNum = endNum // 2
	layer = addFCN(layer, startNum, actualEndNum)
	# now do the max pooling per
	pooling = MaxPoolingVFELayer()(layer)
	pooling = RepeatLayer()(pooling)
	# Copy the layer list to prevent error cycle in concat.
	# https://github.com/tensorflow/tensorflow/issues/30355
	concatLayers = [pooling, layer]
	out = Concatenate()(concatLayers[:])
	return out


def addFCN(layer, startNum, endNum):
	layer = addDenseLayer(layer, endNum)
	layer = BatchNormalization()(layer)
	#layer = addDenseLayer(layer, endNum, 'relu')
	layer = Activation('relu')(layer)
	return layer


# keras dense layers don't support tensor with rank 5 and above, so we need to temporarily reshape
def addDenseLayer(layer, units, act=None):
	oldShape = layer.shape[1:]
	pointShape = layer.shape[-2:]
	combineVoxel = np.prod(np.array(layer.shape[1:-2]))
	shape = (combineVoxel,) + pointShape
	layer = Reshape(shape)(layer)
	layer = Dense(units, input_shape=(layer.shape[1:]), activation=act, use_bias=False)(layer)
	layer = Reshape(oldShape[:-1] + (units,))(layer)
	return layer


# Convolution middle layers.
# k = kernel size, s = stride size, p = padding to add.
def addConv3DLayer(layer, cin, cout, k, s, p):
	layer = ZeroPadding3D(padding=p)(layer)
	layer = Conv3D(cout, kernel_size=k, strides=s, padding='valid')(layer)
	layer = BatchNormalization()(layer)
	layer = addDenseLayer(layer, layer.shape[-1], 'relu')
	return layer


# Convolution2D layers for RPN
# k = kernel size, s = stride size, p = padding to add.
def addConv2DLayer(layer, cin, cout, k, s, p):
	layer = ZeroPadding2D(padding=p)(layer)
	layer = Conv2D(cout, kernel_size=k, strides=s)(layer)
	layer = BatchNormalization()(layer)
	layer = addDenseLayer(layer, layer.shape[-1], 'relu')
	return layer


# Modification to RPN. Do a single layer of step size 2, then do q convolutions of step size 1.
def addRPNConvLayer(layer, cin, cout, q):
	layer = addConv2DLayer(layer, cin, cout, 3, 2, 1)
	for i in range(q):
		layer = addConv2DLayer(layer, cout, cout, 3, 1, 1)
	return layer


def getRPNInputShape(layerShape):
	return layerShape[1:-2] + (layerShape[-2] * layerShape[-1],)


def createModel(nx, ny, nz, maxPoints):
	# Keras time
	os.environ[
		"PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

	# Input is a tensor that separates each voxel. Empty voxels are all 0.
	# VFE layers
	inputShape = (nz, nx, ny, maxPoints, 6)
	inLayer = Input(shape=inputShape, name='InputVoxel')
	outLayer = addVFELayer(inLayer, 6, 32)
	outLayer = addVFELayer(outLayer, 32, 64)
	outLayer = addFCN(outLayer, 64, 64)
	# Convolution layers. Just use default convolution algorithm.
	outLayer = MaxPoolingVFELayer(combine=True)(outLayer)
	outLayer = addConv3DLayer(outLayer, 64, 64, 3, (2, 1, 1), (1, 1, 1))
	outLayer = addConv3DLayer(outLayer, 64, 64, 3, (1, 1, 1), (0, 1, 1))
	outLayer = addConv3DLayer(outLayer, 64, 64, 3, (2, 1, 1), (1, 1, 1))
	# RPN layer time
	# format data so we can run RPN on it and treat it like a 2D image.
	# after each rpbConvLayer, decompose and save for concat at end.
	outLayer = Permute((2, 3, 4, 1))(outLayer)
	outLayer = Reshape(getRPNInputShape(outLayer.shape))(outLayer)
	# block 1
	rpnConv = addRPNConvLayer(outLayer, 128, 128, 3)
	rpnConv1Out = Conv2DTranspose(256, strides=1, kernel_size=3, padding='same')(rpnConv)
	# block 2
	rpnConv = addRPNConvLayer(rpnConv, 128, 128, 5)
	rpnConv2Out = Conv2DTranspose(256, strides=2, kernel_size=2, padding='same')(rpnConv)
	# block 3
	rpnConv = addRPNConvLayer(rpnConv, 128, 256, 5)
	rpnConv3Out = Conv2DTranspose(256, strides=4, kernel_size=4, padding='same')(rpnConv)
	outLayer = Concatenate()([rpnConv1Out, rpnConv2Out, rpnConv3Out])
	probabilityLayer = Conv2D(2, kernel_size=1, strides=1, padding='same', name='ClassificationLayer')(outLayer)
	regressionMap = Conv2D(14, kernel_size=1, strides=1, padding='same', name='RegressionLayer')(outLayer)
	model = Model(inputs=inLayer, outputs=[probabilityLayer, regressionMap])
	return model


def main2(samples):
	import time
	# Set constants
	dataDir = 'C:\\Users\\snkim\\Desktop\\poject\\data'
	labelsDir = 'C:\\Users\\snkim\\Desktop\\poject\\labels_first_sample_threading'
	#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	points = []
	#for sample in samples:
	for i in range(len(samples)):
		# pre-process data
		sampleLidarPoints = combine_lidar_data(samples[i], dataDir)
		startTime = time.time()
		trainVFEPoints = VFE_preprocessing(sampleLidarPoints, voxelx, voxely, voxelz, maxPoints, nx // 2, ny // 2, nz)
		testVFEPointsDense = sparse.to_dense(trainVFEPoints, default_value=0., validate_indices=False)
		points.append(testVFEPointsDense)
		endTime = time.time()
		print(endTime - startTime)
		print('finished ' + str(i))
		# Turn into 6 rank tensor, then convert it to dense because keras is stupid
		# testVFEPoints = sparse.reshape(testVFEPoints, (1,) + testVFEPoints.shape)
		# testVFEPointsDense = sparse.to_dense(testVFEPoints, default_value=0., validate_indices=False)
	trainPoints = tf.stack(points, axis=0)

	print('loading labels')
	# get labels from file 
	outClass = np.load(labelsDir + '\\labelsClass.npy', allow_pickle=True)
	outRegress = np.load(labelsDir + '\\regressClass.npy', allow_pickle=True)
	classShape = np.load(labelsDir + '\\labelsShape.npy', allow_pickle=True)
	regressShape = np.load(labelsDir + '\\regressShape.npy', allow_pickle=True)
	# outClass = outClass.reshape((tuple(classShape)))
	# outRegress = outRegress.reshape((tuple(regressShape)))
	
	# I fucked up the serialize data. It's saved as [id, array object we want]
	outClass = np.stack(outClass[:,1])
	outRegress = np.stack(outRegress[:,1])

	# create model
	model = createModel(nx, ny, nz, maxPoints)
	# plot_model(model, show_shapes=True)
	sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss=['mse', 'mse'])
	#model.summary()
	# model =load_model('models\\Epoch5.h5', custom_objects={'RepeatLayer' : RepeatLayer, 'MaxPoolingVFELayer' : MaxPoolingVFELayer})

	# fit model
	history = model.fit(x=trainPoints, y=[outClass, outRegress], batch_size=1, verbose=1, epochs=1, steps_per_epoch=180)

	print(history.history)
	model.save('C:\\Users\\snkim\\Desktop\\poject\\models\\180SampleEpoch0.h5')


if __name__ == '__main__':
	#scene = level5Data.scene[0]
	#sample = level5Data.get('sample', scene['first_sample_token'])
	#sample2 = level5Data.get('sample', sample['next'])
	#main2([sample, sample2])
	samples = []
	for scene in level5Data.scene:
		samples.append(level5Data.get('sample', scene['first_sample_token']))
	print('Training on ' + str(len(samples)))
	main2(samples[:])
