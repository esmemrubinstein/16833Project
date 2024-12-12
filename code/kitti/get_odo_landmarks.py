import numpy as np
import pykitti

# Change this to the directory where you store KITTI data
BASEDIR = '/home/ethan/coursework/16833/16833Project/kitti_data/'

# Specify the dataset to load
DATE = '2011_09_30'
DRIVE = '0034'

# Specify the max number of landmark measurements to show for each frame
NUM_LANDMARKS = 15

# Number of frames to load
NUM_FRAMES = 1224

# Threshold which indicates same height plane
Z_THRESH = 0.2


def clamp_angle(theta):
    while theta >= np.pi:
        theta -= 2*np.pi

    while theta < -np.pi:
        theta += 2*np.pi

    return theta

# for each frame in the given KITTI dataset object, get the top num_landmarks
# in terms of reflectance and return x, y, z coordinates (velodyne, IMU and global)
def get_landmarks(dataset, num_landmarks):
    landmarks = [np.array([]) for _ in dataset.frames]
    landmarks_imu = [np.array([]) for _ in dataset.frames]
    landmarks_global = [np.array([]) for _ in dataset.frames]
    for idx in dataset.frames:
        # only keep the points around the same height as the sensor
        points = dataset.get_velo(idx)
        points = points[points[:, 2] <= Z_THRESH]
        assert(len(points) >= num_landmarks)

        # take the top num_landmarks points based on reflectance
        sorted_idx = np.argsort(points[:, 3])[::-1]
        sorted_idx = sorted_idx[:num_landmarks]
        sorted_points = points.copy()[sorted_idx]
        landmarks[idx] = sorted_points[:, :3]

        # transform to IMU coordinate frame
        t1 = dataset.calib.T_velo_imu
        t1 = np.linalg.inv(t1)
        sorted_points[:, 3] = 1
        sorted_points = (t1 @ sorted_points.T).T
        landmarks_imu[idx] = sorted_points[:, :3]

        # transform to global coordinate frame
        t2 = dataset.oxts[idx].T_w_imu
        sorted_points /= sorted_points[:, 3].reshape(num_landmarks, 1)
        sorted_points = (t2 @ sorted_points.T).T
        sorted_points /= sorted_points[:, 3].reshape(num_landmarks, 1)

        landmarks_global[idx] = sorted_points[:, :3]
    return (np.array(landmarks), np.array(landmarks_imu), np.array(landmarks_global))

# given landmark set, derive the range and horizontal bearing values to be used in EKF-SLAM
# assume given landmark set as coordinates in the velodyne frame
def get_range_bearing(dataset, landmarks):
    num_landmarks = len(landmarks[0])
    lasers = [np.zeros((num_landmarks, 2)) for _ in dataset.frames]
    for idx in dataset.frames:
        lx = landmarks[idx, :, 0]
        ly = landmarks[idx, :, 1]
        lz = landmarks[idx, :, 2]

        imu = dataset.oxts[idx].T_w_imu[:3, 3]
        imu = np.reshape(np.append(imu, np.ones(1)), (4, 1))
        velo = dataset.calib.T_velo_imu @ imu
        imu = np.reshape(imu, 4)
        imu /= imu[3]

        x, y, z = velo[0], velo[1], velo[2]
        dx = lx - x
        dy = ly - y
        dz = lz - z

        lasers[idx][:, 0] = np.sqrt(dx**2 + dy**2 + dz**2)
        lasers[idx][:, 1] = np.arctan2(dy, dx);
    return np.array(lasers)

# given the KITTI dataset, find the global position of the vehicle in each frame (GPS)
def get_gps(dataset):
    odo = [np.zeros(3) for _ in dataset.frames]
    for idx in dataset.frames:
        odo[idx][0] = dataset.timestamps[idx].timestamp()
        odo[idx][1] = dataset.oxts[idx].T_w_imu[0, 2]
        odo[idx][2] = dataset.oxts[idx].T_w_imu[1, 2]

    return np.array(odo)

# given the KITTI dataset, find the odometry between each of the frames, also prepending 
# with timestamp for each frame/entry
def get_odo(dataset):
    odo = [np.zeros(3) for _ in dataset.frames]
    for idx in dataset.frames:
        if idx == 0:
            t1 = np.eye(3)
        else:
            t1 = dataset.oxts[idx - 1].T_w_imu[:3, :3]
            t1[2, :2] = 0
            t1[2, 2] = 1
            t1[:2, 2] = dataset.oxts[idx - 1].T_w_imu[:2, 3]

        t2 = dataset.oxts[idx].T_w_imu[:3, :3]
        t2[2, :2] = 0
        t2[2, 2] = 1
        t2[:2, 2] = dataset.oxts[idx].T_w_imu[:2, 3]

        th1 = np.arctan2(t1[1, 0], t1[0, 0])
        th2 = np.arctan2(t2[1, 0], t2[0, 0])
        dth = clamp_angle(th2 - th1)

        dx = t2[0, 2] - t1[0, 2]
        dy = t2[1, 2] - t1[1, 2]

        r = np.sqrt(dx**2 + dy**2)

        odo[idx][0] = dataset.timestamps[idx].timestamp()
        odo[idx][1] = r
        odo[idx][2] = dth

    return np.array(odo)

def save_to_file(filename, arr):
    np.savetxt(filename, arr, delimiter=',', fmt = '%.7f')


if __name__ == '__main__':
    dataset = pykitti.raw(BASEDIR, DATE, DRIVE, frames=range(0, NUM_FRAMES))

    landmarks = get_landmarks(dataset, NUM_LANDMARKS)[0]
    print("Got landmarks: " + str(np.shape(landmarks)))
    lasers = get_range_bearing(dataset, landmarks)
    ranges = lasers[:, :, 0]
    ranges = np.concatenate((np.reshape(np.array([dataset.timestamps[idx].timestamp() for idx in dataset.frames]), (NUM_FRAMES, 1)), ranges), axis=1)
    bearings = lasers[:, :, 1]
    bearings = np.concatenate((np.reshape(np.array([dataset.timestamps[idx].timestamp() for idx in dataset.frames]), (NUM_FRAMES, 1)), bearings), axis=1)
    print("Got ranges and bearings: " + str(np.shape(ranges)))

    odo = get_odo(dataset)
    print("Got odometry")

    save_to_file('/home/ethan/coursework/16833/16833Project/kitti_data/{}_{}_gps.txt'.format(DATE, DRIVE), get_gps(dataset))
    save_to_file('/home/ethan/coursework/16833/16833Project/kitti_data/{}_{}_odo.txt'.format(DATE, DRIVE), odo)
    save_to_file('/home/ethan/coursework/16833/16833Project/kitti_data/{}_{}_range.txt'.format(DATE, DRIVE), ranges)
    save_to_file('/home/ethan/coursework/16833/16833Project/kitti_data/{}_{}_bearing.txt'.format(DATE, DRIVE), bearings)