import matplotlib

matplotlib.use('agg')

from lyft_dataset_sdk.lyftdataset import LyftDataset
import numpy as np
import serialize_data_threading as LoadDataModule


def applyRegrssion(x, y, z, tx, ty, tz, theta):
	pass


# Convert RPN matrices for a single sample into list of regions with cars
# labelsClass is shape (100, 200, 2),
# regressClass is shape (100, 200, 14)
def rpnToRegion(labelsClass, labelsRegress):
	# A is the coordinates for the 2 anchors for every point in the feature map
	#	Coordinates are x, y, z, l, w, h, yaw
	A = np.zeros((7,) + labelsClass.shape)

	for i in range(len(LoadDataModule.anchors)):
		currAnchor = LoadDataModule.anchors[i]
		currRegress = labelsRegress[:, :, i * 7:i * 7 + 7]

		# populate A with the 7 coordinates for every anchor
		A[0, :, :, i] = 0
		A[1, :, :, i] = 0
		A[2, :, :, i] = -0.5
		A[3, :, :, i] = currAnchor[0]		# length of anchor
		A[4, :, :, i] = currAnchor[1]		# width of anchor
		A[5, :, :, i] = currAnchor[2]		# height of anchor
		A[6, :, :, i] = currAnchor[3]		# yaw of anchor


if __name__ == '__main__':
	dataDir = 'C:\\Users\\snkim\\Desktop\\poject\\data'

	# load dataset
	level5Data = LyftDataset(
		data_path=dataDir,
		json_path=dataDir + '\\train_data',
		verbose=True
	)
	predictClass = np.load('C:\\Users\\snkim\\Desktop\\poject')
	predictRegress = np.load('C:\\Users\\snkim\\Desktop\\poject')
