import matplotlib

import math
from lyft_dataset_sdk.lyftdataset import LyftDataset
import numpy as np
import serialize_data as LoadDataModule
from model_training import combine_lidar_data
from pyquaternion import Quaternion
from shapely.ops import cascaded_union
from shapely.geometry import Polygon
import Constants

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches


def nonMaxSuppressionFast(boxInfo, probInfo, overlapThresh=0.9, maxBoxes=300):
	# Steps:
	#	Sort probability information
	#	Find largest probabiliy, save as 'Last'
	#	Calculate IoU with 'Last' box and all other boxes in list. If IoU is larger than overlap thresh, delete the box
	#	repeat above 2 steps until no items left in probability information.

	if len(probInfo) == 0:
		return [], []

	xInfo = boxInfo[:, 0]
	yInfo = boxInfo[:, 1]
	zInfo = boxInfo[:, 2]
	lengthInfo = boxInfo[:, 3]
	widthInfo = boxInfo[:, 4]
	heightInfo = boxInfo[:, 5]
	yawInfo = boxInfo[:, 6]

	# list of picked indexes to return
	pick = []

	# sort probabilities
	idxs = np.argsort(probInfo)

	while len(idxs) > 0:
		print('fast max suppression idx length:', len(idxs), 'picks count:', len(pick))
		# get the last (highest prob) value
		last = len(idxs) - 1
		currI = idxs[last]
		pick.append(currI)

		# find iou
		lastBox = [xInfo[currI], yInfo[currI], zInfo[currI],
				   lengthInfo[currI], widthInfo[currI], heightInfo[currI],
				   yawInfo[currI]]
		toDelete = []
		for subI in idxs[:last]:
			if xInfo[subI] - Constants.anchors[0][0] < 0 \
					or xInfo[subI] + Constants.anchors[0][0] > 100 \
					or yInfo[subI] - Constants.anchors[0][1] < 0 \
					or yInfo[subI] + Constants.anchors[0][1] > 100:
				toDelete.append(subI)
			else:
				box = [xInfo[subI], yInfo[subI], zInfo[subI],
					   lengthInfo[subI], widthInfo[subI], heightInfo[subI],
					   yawInfo[subI]]
				iou = LoadDataModule.calculateIoU(lastBox, box)
				if iou > overlapThresh:
					toDelete.append(subI)
		idxs = np.delete(idxs, (last,))
		idxs = np.delete(idxs, toDelete)

		if len(pick) > maxBoxes:
			break
	boxes = boxInfo[pick]
	probs = probInfo[pick]
	return boxes, probs


def applyRegrssion(x, y, z, l, w, h, theta, tx, ty, tz, tl, tw, th, tyaw):
	# x, y, z, l, w, h are references to the anchor box
	# tx, ty, tz, tl, tw, th are the regression values
	box_x = tx * l + x
	box_y = ty * w + y
	box_z = tz * h + z
	box_l = np.exp(tl) * l
	box_w = np.exp(tw) * w
	box_h = np.exp(th) * h
	box_yaw = tyaw + theta
	return box_x, box_y, box_z, box_l, box_w, box_h, box_yaw


def applyRegrssionNP(X, regress):
	# same as apply regression but do it across a lot of values
	# X is anchor values, size is (7, 0utX, OutY)
	# regress is regression (t) values, size is (7, 0utX, OutY)
	x = X[0, :, :]
	y = X[1, :, :]
	z = X[2, :, :]
	l = X[3, :, :]
	w = X[4, :, :]
	h = X[5, :, :]
	theta = X[6, :, :]

	tx = regress[0, :, :]
	ty = regress[1, :, :]
	tz = regress[2, :, :]
	tl = regress[3, :, :]
	tw = regress[4, :, :]
	th = regress[5, :, :]
	tyaw = regress[6, :, :]

	box_x, box_y, box_z, box_l, box_w, box_h, box_yaw \
		= applyRegrssion(x, y, z, l, w, h, theta, tx, ty, tz, tl, tw, th, tyaw)
	return np.stack((box_x, box_y, box_z, box_l, box_w, box_h, box_yaw))


# Convert RPN matrices for a single sample into list of regions with cars
# labelsClass is shape (100, 200, 2),
# regressClass is shape (100, 200, 14)
def rpnToRegion(labelsClass, labelsRegress):
	# ASSUMES THAT OUTPUT OF RPN IS MAP DIVIDED BY 2. Our network does this.
	# Assumes data input is in cm.
	outX = Constants.nx // 2
	outY = Constants.ny // 2
	voxelXSize = Constants.voxelx * 2
	voxelYSize = Constants.voxely * 2

	# A is the coordinates for the 2 anchors for every point in the feature map
	#	Coordinates are x, y, z, l, w, h, yaw
	A = np.zeros((7,) + labelsClass.shape)
	X, Y = np.meshgrid(np.arange(outX), np.arange(outY))

	for i in range(len(Constants.anchors)):
		currAnchor = Constants.anchors[i]
		currRegress = labelsRegress[:, :, i * 7:i * 7 + 7]
		currRegress = np.transpose(currRegress, (2, 0, 1))  # move

		# populate A with the 7 coordinates for every anchor
		A[0, :, :, i] = X.T * voxelXSize + voxelXSize / 2
		A[1, :, :, i] = Y.T * voxelYSize + voxelYSize / 2
		A[2, :, :, i] = 1.
		A[3, :, :, i] = currAnchor[0]  # length of anchor
		A[4, :, :, i] = currAnchor[1]  # width of anchor
		A[5, :, :, i] = currAnchor[2]  # height of anchor
		A[6, :, :, i] = currAnchor[3]  # yaw of anchor

		# calculae regression
		A[:, :, :, i] = applyRegrssionNP(A[:, :, :, i], currRegress)

	# becomes 1D array of (100*200 + 100*200,) where i is grouped by anchror
	probInfo = labelsClass.transpose((2, 0, 1)).reshape((-1))
	# becomes 2D array where each row is the 7 numbers for the box.
	boxInfo = np.reshape(A.transpose((0, 3, 1, 2)), (7, -1)).transpose((1, 0))

	lengthInfo = boxInfo[:, 3]
	widthInfo = boxInfo[:, 4]
	heightInfo = boxInfo[:, 5]

	# remove illegal boxes
	idxs = np.where((lengthInfo < 0) | (widthInfo < 0) | (heightInfo < 0))
	if (len(idxs[0]) > 0):
		boxInfo = np.delete(boxInfo, idxs, 0)
		probInfo = np.delete(probInfo, idxs, 0)

	result = nonMaxSuppressionFast(boxInfo, probInfo, maxBoxes=20, overlapThresh=0.)
	return result


def showAnn(sample, plot):
	annsTokens = sample['anns']
	my_sample_data = level5Data.get('sample_data', sample['data']['LIDAR_TOP'])
	ego = level5Data.get('ego_pose', my_sample_data['ego_pose_token'])
	labels = []
	for token in annsTokens:
		ann = level5Data.get('sample_annotation', token)

		# do a inverse transpose to get the annotation data from global coords to local
		translation = np.array(ann['translation']).reshape((1, -1))
		translation = translation - np.array(ego['translation'])
		translation = LoadDataModule.rotate_points(translation, np.array(ego['rotation']), True)

		row = []
		row += [translation[0, 0]]
		row += [translation[0, 1]]
		row += [translation[0, 2]]
		row += ann['size']
		quaternion = Quaternion(ann['rotation'])
		row += [quaternion.yaw_pitch_roll[0]]
		instance = level5Data.get('instance', ann['instance_token'])
		category = level5Data.get('category', instance['category_token'])['name']
		# row += [catToNum[category]]
		# Only adds cars within our range of -50 to 50 in x and y
		if category == 'car' \
				and row[0] >= -50 and row[0] <= 50 \
				and row[1] >= -50 and row[1] <= 50:
			labels.append(row)
	labels = np.array(labels)
	plot.scatter(labels[:,0], labels[:,1],s=4, c='#ff0055')
	for box in labels:
		p = patches.Rectangle((box[0] - box[3] / 2, box[1] - box[4] / 2), box[3], box[4], fill=False, color="#ff0055")
		ax.add_patch(p)


def calcIntersectAll(boxBoxes, annsBoxes):
	# Turn each group of boxes into polygons, combine them, then find intersection
	predictPolygons = []
	for box in boxBoxes:
		predictPolygons.append(LoadDataModule.boxToShapely(box))
	predictCombined = cascaded_union(predictPolygons[:])

	labelBoxes = []
	for box in annsBoxes:
		labelBoxes.append(LoadDataModule.boxToShapely(box))
	labelCombined = cascaded_union(labelBoxes[:])
	return predictCombined.intersection(labelCombined).area

def calcUnionAll(boxesBoxes, annsBoxes, intersect):
	annsSum = 0
	for box in annsBoxes:
		annsSum += box[3] * box[4] * box[5]
	predictSum = 0
	for box in boxesBoxes:
		predictSum += box[3] * box[4] * box[5]
	return annsSum + predictSum - intersect

def calcIoUAll(predictBoxes, sample):
	annsTokens = sample['anns']
	my_sample_data = level5Data.get('sample_data', sample['data']['LIDAR_TOP'])
	ego = level5Data.get('ego_pose', my_sample_data['ego_pose_token'])
	labels = []
	for token in annsTokens:
		ann = level5Data.get('sample_annotation', token)
		# do a inverse transpose to get the annotation data from global coords to local
		translation = np.array(ann['translation']).reshape((1, -1))
		translation = translation - np.array(ego['translation'])
		translation = LoadDataModule.rotate_points(translation, np.array(ego['rotation']), True)

		row = []
		row += [translation[0, 0]]
		row += [translation[0, 1]]
		row += [translation[0, 2]]
		row += ann['size']
		quaternion = Quaternion(ann['rotation'])
		row += [quaternion.yaw_pitch_roll[0]]
		instance = level5Data.get('instance', ann['instance_token'])
		category = level5Data.get('category', instance['category_token'])['name']
		# row += [catToNum[category]]
		# Only adds cars within our range of -50 to 50 in x and y
		if category == 'car' \
				and row[0] >= -50 and row[0] <= 50 \
				and row[1] >= -50 and row[1] <= 50:
			labels.append(row)
	labelsBoxes = np.array(labels)

	intersect = calcIntersectAll(predictBoxes, labelsBoxes)
	union = calcUnionAll(predictBoxes, labelsBoxes, intersect)
	return intersect / union


if __name__ == '__main__':
	# code adapated from 2D RPN to ROI calcultion found at:
	# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	# dataDir = 'C:\\Users\\snkim\\Desktop\\poject\\data'
	dataDir = 'E:\\CS539 Machine Learning\\3d-object-detection-for-autonomous-vehicles'

	# load dataset
	level5Data = LyftDataset(
		data_path=dataDir,
		json_path=dataDir + '\\train_data',
		verbose=True
	)
	predictClass = np.load('fixedTheta\\sample2_label.npy')
	predictRegress = np.load('fixedTheta\\sample2_regress.npy')

	predictClass = np.reshape(predictClass, predictClass.shape[1:])
	predictRegress = np.reshape(predictRegress, predictRegress.shape[1:])

	boxes, probs = rpnToRegion(predictClass, predictRegress)

	# fix positioning on boxes
	boxes[:, 0] = boxes[:, 0] - 50
	boxes[:, 1] = boxes[:, 1] - 50

	# now lets do some checking
	sample = level5Data.get('sample', level5Data.scene[2]['first_sample_token'])
	lidarPoints = combine_lidar_data(sample, dataDir, level5Data)
	fig = plt.figure(figsize=(12, 12))
	ax = fig.add_subplot(111)
	# plt.axis('equal')
	ax.scatter(np.clip(lidarPoints[:, 0], -50, 50),
			   np.clip(lidarPoints[:, 1], -50, 50), s=1, c='#000000')
	ax.scatter(boxes[:, 0], boxes[:, 1], s=4, c='#3461eb')
	for box in boxes:
		p = patches.Rectangle((box[0] - box[3] / 2, box[1] - box[4] / 2), box[3], box[4], fill=False, color='#3461eb')
		ax.add_patch(p)
	showAnn(sample, ax)
	print('IoU for sample:',calcIoUAll(boxes, sample))
	plt.show()
