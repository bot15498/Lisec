import math

# Directory for Lyft dataset
lyft_data_dir = 'E:\\CS539 Machine Learning\\3d-object-detection-for-autonomous-vehicles'

# size of voxel
voxelx = 0.5
voxely = 0.25
voxelz = 0.25

# Number of voxels in space that we care about
nx = int(100 / voxelx)  # -50 to 50 m
ny = int(100 / voxely)
nz = int(2 / voxelz)

# number of anchors
anchors = [[1.6, 3.9, 1.56, 0], [1.6, 3.9, 1.56, math.pi / 2]]

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

# ============================
# RPN Constants

# region limiter
maxRegions = 256
iouLowerBound = 0.45
iouUpperBound = 0.6
