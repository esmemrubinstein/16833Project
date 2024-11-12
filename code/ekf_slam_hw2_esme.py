'''
    Initially written by Ming Hsiao in MATLAB
    Adapted to Python by Akash Sharma (akashsharma@cmu.edu), 2020
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)


def draw_cov_ellipse(mu, cov, color):
    """
    Draws an ellipse in plt canvas.

    \param mu Mean of a Gaussian
    \param cov Covariance of a Gaussian
    \param color Color in plt format, e.g. 'b' for blue, 'r' for red.
    """ 
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0, 1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2 * np.pi, np.pi / 50)
    rot = []
    for i in range(100):
        rect = (np.array(
            [3 * np.sqrt(a) * np.cos(phi[i]),
             3 * np.sqrt(b) * np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + mu)

    rot = np.asarray(rot)
    plt.plot(rot[:, 0], rot[:, 1], c=color, linewidth=0.75)


def draw_traj_and_pred(X, P):
    """ Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'm')
    plt.draw()
    plt.waitforbuttonpress(0)


def draw_traj_and_map(X, last_X, P, t):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'b')
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c='b', linewidth=0.75)
    plt.plot(X[0], X[1], '*b')

    if t == 0:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + k * 2:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'r')
    else:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + 2 * k:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'g')

    plt.draw()
    plt.waitforbuttonpress(0)


def warp2pi(angle_rad):
    """
    TODO: warps an angle in [-pi, pi]. Used in the update step.

    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    angle_rad = angle_rad + np.pi
    angle_rad = angle_rad % (2 * np.pi)
    angle_rad = angle_rad - np.pi

    return angle_rad


def init_landmarks(init_measure, init_measure_cov, init_pose, init_pose_cov):
    '''
    TODO: initialize landmarks given the initial poses and measurements with their covariances
    \param init_measure Initial measurements in the form of (beta0, l0, beta1, l1, ...).
    \param init_measure_cov Initial covariance matrix of shape (2, 2) per landmark given parameters.
    \param init_pose Initial pose vector of shape (3, 1).
    \param init_pose_cov Initial pose covariance of shape (3, 3) given parameters.

    \return k Number of landmarks.
    \return landmarks Numpy array of shape (2k, 1) for the state.
    \return landmarks_cov Numpy array of shape (2k, 2k) for the uncertainty.
    '''

    k = init_measure.shape[0] // 2

    landmark = np.zeros((2 * k, 1))
    landmark_cov = np.zeros((2 * k, 2 * k))

    x_t, y_t, theta_t = init_pose.flatten()

    for i in range(k):
        beta = init_measure[2 * i].item()
        r = init_measure[2 * i + 1].item()

        l_x = x_t + r * np.cos(theta_t + beta)
        l_y = y_t + r * np.sin(theta_t + beta)
        
        landmark[2 * i] = l_x
        landmark[2 * i + 1] = l_y

        # Jacobian of landmark pose w.r.t. robot pose
        jacobian_p = np.array([[1, 0, -r * np.sin(theta_t + beta)],
                               [0, 1, r * np.cos(theta_t + beta)]])
        
        # Jacobian of measurement w.r.t. landmark pose
        jacobian_m = np.array([[-r * np.sin(theta_t + beta), np.cos(theta_t + beta)],
                               [r * np.cos(theta_t + beta), np.sin(theta_t + beta)]])
        
        # 1.2 in Theory section
        landmark_cov[2 * i:2 * i + 2, 2 * i:2 * i + 2] = jacobian_p @ init_pose_cov @ jacobian_p.T + jacobian_m @ init_measure_cov @ jacobian_m.T
        

    return k, landmark, landmark_cov


def predict(X, P, control, control_cov, k):
    '''
    TODO: predict step in EKF SLAM with derived Jacobians.
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.
    \param control Control signal of shape (2, 1) in the polar space that moves the robot.
    \param control_cov Control covariance of shape (3, 3) in the (x, y, theta) space given the parameters.
    \param k Number of landmarks.

    \return X_pre Predicted X state of shape (3 + 2k, 1).
    \return P_pre Predicted P covariance of shape (3 + 2k, 3 + 2k).
    '''

    x_t, y_t, theta_t = X[0:3].flatten()

    d_t = control[0]
    alpha_t = control[1]

    # Incorporate control signal
    x_t1 = x_t + d_t * np.cos(theta_t)
    y_t1 = y_t + d_t * np.sin(theta_t)
    theta_t1 = warp2pi(theta_t + alpha_t)

    X[:3] = np.array([[x_t1], [y_t1], [theta_t1]]).reshape(3, 1)

    H_p = np.eye(3 + 2 * k) # motion model jacobian
    H_p[0,2] = -d_t * np.sin(theta_t)
    H_p[1,2] = d_t * np.cos(theta_t)

    H_c = np.zeros((3, 3)) # control noise jacobian
    H_c[0, 0] = np.cos(theta_t)   
    H_c[0, 1] = -np.sin(theta_t)  
    H_c[1, 0] = np.sin(theta_t)        
    H_c[1, 1] = np.cos(theta_t)   
    H_c[2, 2] = 1                     

    H_c_extended = np.zeros((15, H_c.shape[1]))
    H_c_extended[:3, :] = H_c

    # 4. Update covariance
    P = H_p @ P @ H_p.T + H_c_extended @ control_cov @ H_c_extended.T

    return X, P


def update(X_pre, P_pre, measure, measure_cov, k):
    '''
    Update step in EKF SLAM with derived Jacobians.
    
    :param X_pre: Predicted state vector of shape (3 + 2k, 1) from the predict step.
    :param P_pre: Predicted covariance matrix of shape (3 + 2k, 3 + 2k) from the predict step.
    :param measure: Measurement signal of shape (2k, 1).
    :param measure_cov: Measurement covariance of shape (2, 2) per landmark given the parameters.
    :param k: Number of landmarks.

    :return: X Updated X state of shape (3 + 2k, 1).
    :return: P Updated P covariance of shape (3 + 2k, 3 + 2k).
    '''
    
    X = X_pre.copy()
    P = P_pre.copy()

    # Initialize the Jacobian H and expected measurements z hat
    H = np.zeros((2 * k, 3 + 2 * k))
    z_hat = np.zeros((2 * k, 1))

    for i in range(k):
        l_x = X_pre[3 + 2 * i]
        l_y = X_pre[3 + 2 * i + 1]


        x_t, y_t, theta_t = X_pre[0:3].flatten()

        delta_x = l_x - x_t
        delta_y = l_y - y_t

        c = np.sqrt(delta_x**2 + delta_y**2)

        z_hat[2 * i] = warp2pi(np.arctan2(delta_y, delta_x) - theta_t)
        z_hat[2 * i + 1] = c

        # Fill the Jacobian H - bearing angle is FIRST
        H[2 * i + 1, 0] = -delta_x / c  # ∂r/∂x
        H[2 * i + 1, 1] = -delta_y / c  # ∂r/∂y
        H[2 * i + 1, 3 + 2 * i] = delta_x / c  # ∂r/∂l_x
        H[2 * i + 1, 3 + 2 * i + 1] = delta_y / c  # ∂r/∂l_y

        H[2 * i, 0] = delta_y / (c**2)  # ∂β/∂x
        H[2 * i, 1] = -delta_x / (c**2)  # ∂β/∂y
        H[2 * i, 2] = -1  # ∂β/∂θ
        H[2 * i, 3 + 2 * i] = -delta_y / (c**2)  # ∂β/∂l_x
        H[2 * i, 3 + 2 * i + 1] = delta_x / (c**2)  # ∂β/∂l_y

    # extend measurement cov to each landmark
    measure_cov_full = block_diag(*[measure_cov for _ in range(k)])
    # Kalman gain
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + measure_cov_full)
    X = X + K @ (measure - z_hat)
    P = (np.eye(3 + 2 * k) - K @ H) @ P

    return X, P


def euclidean_distance(l_ekf, l_true):
    dist = np.sqrt((l_ekf[0] - l_true[0])**2 + (l_ekf[1] - l_true[1])**2)
    return dist


def mahalanobis_distance(l_ekf, l_true, cov):
    diff = l_ekf - l_true
    dist = np.sqrt(diff.T @ np.linalg.inv(cov) @ diff)
    return dist


def evaluate(X, P, k):
    '''
    TODO: evaluate the performance of EKF SLAM.
    1) Plot the results.
    2) Compute and print the Euclidean and Mahalanobis distance given X, P, and the ground truth (provided in the function).
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.

    \return None
    '''
    l_true = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=float)
    plt.scatter(l_true[0::2], l_true[1::2])
    plt.draw()
    plt.waitforbuttonpress(0)

    euclidean_distances = []
    mahalanobis_distances = []
    for i in range(k):
        l_ekf = X[3 + 2 * i:3 + 2 * i + 2].flatten()
        landmark_cov = P[3 + 2 * i:3 + 2 * i + 2, 3 + 2 * i:3 + 2 * i + 2]
        l_true_i = l_true[2 * i:2 * i + 2]
        euclidean_distances.append(euclidean_distance(l_ekf, l_true_i))
        mahalanobis_distances.append(mahalanobis_distance(l_ekf, l_true_i, landmark_cov))

    print(f'Euclidean distances: {euclidean_distances}')
    print(f'Mahalanobis distances: {mahalanobis_distances}')


def main():
    # TEST: Setup uncertainty parameters
    sig_x = 0.25
    sig_y = 0.1
    sig_alpha = 0.1
    sig_beta = 0.01
    sig_r = 0.08


    # Generate variance from standard deviation
    sig_x2 = sig_x**2
    sig_y2 = sig_y**2
    sig_alpha2 = sig_alpha**2
    sig_beta2 = sig_beta**2
    sig_r2 = sig_r**2

    # Open data file and read the initial measurements
    data_file = open("data/data.txt")
    line = data_file.readline()
    fields = re.split('[\t ]', line)[:-1]
    arr = np.array([float(field) for field in fields])
    measure = np.expand_dims(arr, axis=1)
    t = 1

    # Setup control and measurement covariance
    control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
    measure_cov = np.diag([sig_beta2, sig_r2])

    # Setup the initial pose vector and pose uncertainty
    pose = np.zeros((3, 1))
    pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

    ##########
    # TODO: initialize landmarks
    k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose,
                                               pose_cov)

    # Setup state vector X by stacking pose and landmark states
    # Setup covariance matrix P by expanding pose and landmark covariances
    X = np.vstack((pose, landmark))
    P = np.block([[pose_cov, np.zeros((3, 2 * k))],
                  [np.zeros((2 * k, 3)), landmark_cov]])

    # Plot initial state and covariance
    last_X = X
    draw_traj_and_map(X, last_X, P, 0)

    # Core loop: sequentially process controls and measurements
    for line in data_file:
        fields = re.split('[\t ]', line)[:-1]
        arr = np.array([float(field) for field in fields])

        # Control
        if arr.shape[0] == 2:
            print(f'{t}: Predict step')
            d, alpha = arr[0], arr[1]
            control = np.array([[d], [alpha]])

            ##########
            # TODO: predict step in EKF SLAM
            X_pre, P_pre = predict(X, P, control, control_cov, k)

            draw_traj_and_pred(X_pre, P_pre)

        # Measurement
        else:
            print(f'{t}: Update step')
            measure = np.expand_dims(arr, axis=1)

            ##########
            # TODO: update step in EKF SLAM
            X, P = update(X_pre, P_pre, measure, measure_cov, k)

            draw_traj_and_map(X, last_X, P, t)
            last_X = X
            t += 1

    # EVAL: Plot ground truth landmarks and analyze distances
    evaluate(X, P, k)


if __name__ == "__main__":
    main()
