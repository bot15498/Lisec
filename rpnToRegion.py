import matplotlib

matplotlib.use('agg')

import math
from lyft_dataset_sdk.lyftdataset import LyftDataset
import numpy as np
import serialize_data_threading as LoadDataModule


def nonMaxSuppressionFast(boxInfo, probInfo, overlapThresh=0.9, maxBoxes=300):
	# Steps:
	#	Sort probability information
	#	Find largest probabiliy, save as 'Last'
	#	Calculate IoU with 'Last' box and all other boxes in list. If IoU is larger than overlap thresh, delete the box
	#	repeat above 2 steps until no items left in probability information.

	if len(probInfo) == 0:
		return []

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
		# get the last (highest prob) value
		last = len(idxs) - 1
		currI = idxs[last]
		pick.append(currI)

		# find iou
		lastBox = [xInfo[currI], yInfo[currI], zInfo[currI],
				   lengthInfo[currI], widthInfo[currI], heightInfo[currI],
				   yawInfo[currI]]
		toDelete = []
		for subI in len(idxs[:last]):
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
	return boxInfo, probInfo


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
	outX = LoadDataModule.nx // 2
	outY = LoadDataModule.ny // 2
	voxelXSize = LoadDataModule.voxelx * 2
	voxelYSize = LoadDataModule.voxely * 2

	# A is the coordinates for the 2 anchors for every point in the feature map
	#	Coordinates are x, y, z, l, w, h, yaw
	A = np.zeros((7,) + labelsClass.shape)
	X, Y = np.meshgrid(np.arange(outX), np.arange(outY, -1, -1))

	for i in range(len(LoadDataModule.anchors)):
		currAnchor = LoadDataModule.anchors[i]
		currRegress = labelsRegress[:, :, i * 7:i * 7 + 7]
		currRegress = np.transpose(currRegress, (2, 0, 1))  # move

		# populate A with the 7 coordinates for every anchor
		A[0, :, :, i] = X * voxelXSize
		A[1, :, :, i] = Y * voxelYSize
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
	boxInfo = np.delete(boxInfo, idxs, 0)
	probInfo = np.delete(probInfo, idxs, 0)

	result = nonMaxSuppressionFast(boxInfo, probInfo)
	return result


if __name__ == '__main__':
	# code adapated from 2D RPN to ROI calcultion found at:
	# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	dataDir = 'C:\\Users\\snkim\\Desktop\\poject\\data'

	# load dataset
	level5Data = LyftDataset(
		data_path=dataDir,
		json_path=dataDir + '\\train_data',
		verbose=True
	)
	predictClass = np.load('C:\\Users\\snkim\\Desktop\\poject')
	predictRegress = np.load('C:\\Users\\snkim\\Desktop\\poject')
