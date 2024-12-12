import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from get_odo_landmarks import get_landmarks
import numpy as np

import pykitti

# Change this to the directory where you store KITTI data
BASEDIR = '/home/ethan/coursework/16833/16833Project/kitti_data/'

# Specify the dataset to load
DATE = '2011_09_30'
DRIVE = '0034'

# Specify the max number of landmark measurements to show for each frame
NUM_LANDMARKS = 15

# Load the data. Optionally, specify the frame range to load.
NUM_FRAMES = 1224
dataset = pykitti.raw(BASEDIR, DATE, DRIVE, frames=range(0, NUM_FRAMES))

# dataset.calib:         Calibration data are accessible as a named tuple
# dataset.timestamps:    Timestamps are parsed into a list of datetime objects
# dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
# dataset.camN:          Returns a generator that loads individual images from camera N
# dataset.get_camN(idx): Returns the image from camera N at idx
# dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
# dataset.get_gray(idx): Returns the monochrome stereo pair at idx
# dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
# dataset.get_rgb(idx):  Returns the RGB stereo pair at idx
# dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
# dataset.get_velo(idx): Returns the velodyne scan at idx

np.set_printoptions(precision=4, suppress=True)
print('\nDrive: ' + str(dataset.drive))
print('\nFrame range: ' + str(dataset.frames))
print('\nIMU-to-Velodyne transformation:\n' + str(dataset.calib.T_velo_imu))
print('\nFirst timestamp: ' + str(dataset.timestamps[0]))

f1 = plt.figure()
ax1 = f1.add_subplot(111)

landmarks = get_landmarks(dataset, NUM_LANDMARKS)[2]
for idx in dataset.frames:
    sorted_points = landmarks[idx]
    ax1.scatter(sorted_points[:, 0],
                sorted_points[:, 1],
                c=np.ones(NUM_LANDMARKS),
                cmap='gray',
                marker='+')
landmarks = np.array(landmarks)
print(np.shape(landmarks))

coords = np.array([dataset.oxts[idx].T_w_imu[:3, 3] for idx in dataset.frames])
print(coords.shape)
ax1.scatter(coords[:, 0],
            coords[:, 1])
ax1.set_title('Ground Truth Trajectory and Landmarks')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

plt.show()