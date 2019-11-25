from lyft_dataset_sdk.lyftdataset import LyftDataset
import os
from tensorflow import SparseTensor, sparse
from pyquaternion import Quaternion
import numpy as np
from math import floor
import math
import tensorflow as tf
from shapely.geometry import Polygon

# constants
# size of voxel
voxelx = 0.5
voxely = 0.25
voxelz = 0.25

# Number of voxels in space that we care about
nx = int(100 / voxelx)  # -50 to 50 m
ny = int(100 / voxely)
nz = int(2 / voxelz)

# Limit of points per voxel to reduce size of data.
maxPoints = 35

# number of anchors
anchors = [[1.6, 3.9, 1.56], [3.9, 1.6, 1.56]]
iouLowerBound = 0.45
iouUpperBound = 0.6

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

dataDir = 'E:\\CS539 Machine Learning\\3d-object-detection-for-autonomous-vehicles'


# Uses quaternions to rotate all points in a scene to match the location of the lidar sensor on the car.
def rotate_points(points, rotation, inverse=False):
	quaternion = Quaternion(rotation)
	if inverse:
		quaternion = quaternion.inverse
	return np.dot(quaternion.rotation_matrix, points.T).T


# Takes the sample dict and returns an array of n,3 with every point in the sample.
def combine_lidar_data(sample, dataDir):
	sensorTypes = ['LIDAR_TOP', 'LIDAR_FRONT_RIGHT', 'LIDAR_FRONT_LEFT']
	sensorFrameMetadata = [level5Data.get('sample_data', sample['data'][x]) for x in sensorTypes]
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


def calculateIntersection(box1, box2):
	# create shapely polygons and find intersection.
	box1P = boxToShapely(box1)
	box2P = boxToShapely(box2)
	area = box1P.intersection(box2P).area
	# find greates lower bound of z and lowest upper bound, then multiply.
	botZ = max(box1[2] - box1[5], box2[2] - box2[5])
	topZ = min(box1[2] + box1[5], box2[2] + box2[5])
	return (topZ - botZ) * area


def boxToShapely(box):
	theta = math.radians(box[6])
	length = box[3]
	width = box[4]
	refPointRight = (box[0] + math.cos(theta) * (width / 2), box[1] - math.sin(theta) * (width / 2))
	refPointLeft = (box[0] - math.cos(theta) * (width / 2), box[1] + math.sin(theta) * (width / 2))
	# for these points switch cos and sin to represent doing it on 90 - theta
	topRight = [refPointRight[0] + math.sin(theta) * (length / 2), refPointRight[1] + math.cos(theta) * (length / 2)]
	botRight = [refPointRight[0] - math.sin(theta) * (length / 2), refPointRight[1] - math.cos(theta) * (length / 2)]
	topLeft = [refPointLeft[0] + math.sin(theta) * (length / 2), refPointLeft[1] + math.cos(theta) * (length / 2)]
	botLeft = [refPointLeft[0] - math.sin(theta) * (length / 2), refPointLeft[1] - math.cos(theta) * (length / 2)]
	return Polygon([topRight, botRight, botLeft, topLeft])


def calculateUnion(box1, box2, intersect):
	box1Vol = box1[3] * box1[4] * box1[5]
	box2Vol = box2[3] * box2[4] * box2[5]
	return box1Vol + box2Vol - intersect


def calculateIoU(box1, box2):
	'''
	Calculate Intersection over union for two boxes
	:param box1: the annotations row for first box. Should be in form <x, y, z, l, w, h, yaw>
	:param box2: the annotations row for second box
	:return: IoU value
	'''
	intersect = calculateIntersection(box1, box2)
	union = calculateUnion(box1, box2, intersect)
	return intersect / union


def fixBoxScaling(dataSize, newX, newY, origX, origY):
	out = np.ones(dataSize)
	out = out.transpose()
	out[3] = [newX / origX for i in range(len(out[3]))]
	out[4] = [newY / origY for i in range(len(out[4]))]
	return out.transpose()


def preprocessLabels(data):
	# ASSUMES THAT OUTPUT OF RPN IS MAP DIVIDED BY 2. Our network does this.
	# Assumes data input is in cm.
	outX = nx // 2
	outY = ny // 2
	voxelXSize = voxelx * 2 * 100
	voxelYSize = voxely * 2 * 100
	# size is based off nx and ny (just divide by 2). 7 comes from x, y, z, l ,w ,h, yaw
	outRegress = np.zeros((outX, outY, len(anchors) * 7))
	outClass = np.zeros((outX, outY, len(anchors)))
	outClassCheck = np.zeros((outX, outY, len(anchors)))  # used to keep track of past IoUs

	# scale back l and w because we cut the size of the feature space by 2 through our network
	fixedData = data * fixBoxScaling(data.shape, outX, outY, nx, ny)

	# Iterate through anchors and bounding boxes in fixedData and update outRegress and outClass as necessary based on IoU
	centerZ = -0.5  # hard set z center of anchors to -.05 dude just trust me.
	for i in range(len(anchors)):
		for xVoxel in range(outX):
			# Do calculations in terms of cm now.
			centerX = voxelXSize * xVoxel + (voxelXSize / 2)
			if centerX - (anchors[i][0] / 2) < 0 or centerX + (anchors[i][0] / 2) > voxelXSize * outX:
				continue
			for yVoxel in range(outY):
				centerY = voxelYSize * yVoxel + (voxelYSize / 2)
				if centerY - (anchors[i][1] / 2) < 0 or centerY + (anchors[i][1] / 2) > voxelYSize * outY:
					continue
				# if we get here, then the anchor is within the range of the area we want to look at
				# now look at every bounding box for best IoU
				for box in fixedData:
					# Create anchorbox representation using set anchro sizes and add 0 for yaw.
					anchorBox = [centerX, centerY, centerZ] + anchors[i] + [0]
					iou = calculateIoU(anchorBox, box)
					if iou > iouUpperBound:
						outClass[xVoxel, yVoxel, i] = 2
					elif iou < iouLowerBound:
						# assumes all anchors start as 'neg'
						outClass[xVoxel, yVoxel, i] = 0
					else:
						outClass[xVoxel, yVoxel, i] = 1
					if iou > outClassCheck[xVoxel, yVoxel, i]:
						# TODO put the update to the regression mapping here (I think)
						outClassCheck[xVoxel, yVoxel, i] = iou
	return [outClass, outRegress]


def rpnToImage(classData, regressData):
	'''
	Converts output of neural net into bounding boxes.
	:param classData:
	:param regressData:
	:return:
	'''
	pass


def saveLabelsForSample(samples, outPath):
	'''
	Converts Lidar data from a sample into rpn form. Saves it as a npy file
	:param samples:
	:param outPath:
	:return:
	'''
	classMap = []
	regressMap = []
	for sample in samples:
		labels = []
		annsTokens = sample['anns']
		for token in annsTokens:
			ann = level5Data.get('sample_annotation', token)
			row = ann['translation']
			row += ann['size']
			quaternion = Quaternion(ann['rotation'])
			row += [quaternion.yaw_pitch_roll[0]]
			instance = level5Data.get('instance', ann['instance_token'])
			category = level5Data.get('category', instance['category_token'])['name']
			row += [catToNum[category]]
			labels.append(row)
		labels = np.array(labels)
		outClass, outRegress = preprocessLabels(labels)
		outClass = np.reshape(outClass, (1,) + outClass.shape)
		outRegress = np.reshape(outRegress, (1,) + outRegress.shape)
		classMap.append(outClass)
		regressMap.append(outRegress)
	classMap = np.stack(classMap, 0)
	regressMap = np.stack(regressMap, 0)
	np.save(outPath + '\\labelsClass.npy', classMap)
	np.save(outPath + '\\regressClass.npy', regressMap)


def saveTrainDataForSample(samples):
	# TODO make this write tensors to file.
	tensors = []
	for sample in samples:
		# pre-process data
		import time
		testSampleLidarPoints = combine_lidar_data(sample, dataDir)
		startTime = time.time()
		testVFEPoints = VFE_preprocessing(testSampleLidarPoints, voxelx, voxely, voxelz,
										  maxPoints, nx // 2, ny // 2, nz)
		endTime = time.time()
		print(endTime - startTime)
		print(testVFEPoints.shape)
		# Turn into 6 rank tensor, then convert it to dense because keras is stupid
		testVFEPoints = sparse.reshape(testVFEPoints, (1,) + testVFEPoints.shape)
		testVFEPointsDense = sparse.to_dense(testVFEPoints, default_value=0., validate_indices=False)
		tensors.append(testVFEPointsDense)
	# return tf.stack(tensors)
	tf.stack(tensors)


if __name__ == '__main__':
	# load dataset
	level5Data = LyftDataset(
		data_path=dataDir,
		json_path=dataDir + '\\train_data',
		verbose=True
	)
	scene = level5Data.scene[0]
	sample = level5Data.get('sample', scene['first_sample_token'])
	sample2 = level5Data.get('sample',sample['next'])
	saveLabelsForSample([sample], 'labels')
