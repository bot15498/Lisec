from lyft_dataset_sdk.lyftdataset import LyftDataset
from tensorflow.keras.models import load_model
from tensorflow import SparseTensor, sparse
from model_training import RepeatLayer, MaxPoolingVFELayer, combine_lidar_data, VFE_preprocessing
import numpy as np
import Constants


def predictMain(samples, outPath, level5Data, model):
	import time
	# Set constants
	dataDir = 'E:\\CS539 Machine Learning\\3d-object-detection-for-autonomous-vehicles'
	# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	points = []
	# for sample in samples:
	for i in range(len(samples)):
		# pre-process data
		sampleLidarPoints = combine_lidar_data(samples[i], dataDir, level5Data)
		startTime = time.time()
		trainVFEPoints = VFE_preprocessing(sampleLidarPoints,
										   Constants.voxelx,
										   Constants.voxely,
										   Constants.voxelz,
										   Constants.maxPoints,
										   Constants.nx // 2,
										   Constants.ny // 2,
										   Constants.nz)
		trainVFEPoints = sparse.reshape(trainVFEPoints, (1,) + trainVFEPoints.shape)
		testVFEPointsDense = sparse.to_dense(trainVFEPoints, default_value=0., validate_indices=False)
		# points.append(testVFEPointsDense)
		endTime = time.time()
		print(endTime - startTime)
		print('finished ' + str(i))
		# Turn into 6 rank tensor, then convert it to dense because keras is stupid
		# testVFEPoints = sparse.reshape(testVFEPoints, (1,) + testVFEPoints.shape)
		# testVFEPointsDense = sparse.to_dense(testVFEPoints, default_value=0., validate_indices=False)
		prob, regress = model.predict(testVFEPointsDense)
		np.save(outPath + '\\sample' + str(i) + '_label.npy', prob)
		np.save(outPath + '\\sample' + str(i) + '_regress.npy', regress)


if __name__ == '__main__':
	# load dataset
	level5Data = LyftDataset(
		data_path=Constants.lyft_data_dir,
		json_path=Constants.lyft_data_dir + '\\train_data',
		verbose=True
	)

	model = load_model('fixedTheta\\15SampleEpoch0_fixed.h5',
					   custom_objects={'RepeatLayer': RepeatLayer, 'MaxPoolingVFELayer': MaxPoolingVFELayer})

	# load data, then call predict
	samples = []
	for scene in level5Data.scene:
		samples.append(level5Data.get('sample', scene['first_sample_token']))
	print('Testing on ' + str(len(samples)))
	predictMain(samples, 'fixedTheta', level5Data, model)
