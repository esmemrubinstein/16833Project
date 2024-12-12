from __future__ import division
import numpy as np
import slam_utils
import tree_extraction
from scipy.stats.distributions import chi2


def motion_model(u, dt, ekf_state, vehicle_params):
    '''
    Computes the discretized motion model for the given vehicle as well as its Jacobian

    Returns:
        f(x,u), a 3x1 vector corresponding to motion x_{t+1} - x_t given the odometry u.

        df/dX, the 3x3 Jacobian of f with respect to the vehicle state (x, y, phi)
    '''

    v_e = u[0].copy()
    alpha = u[1].copy()
    H = vehicle_params['H']
    L = vehicle_params['L']
    a = vehicle_params['a']
    b = vehicle_params['b']
    v_c = (v_e)/(1-np.tan(alpha)*(H/L))
    t_st = ekf_state['x'].copy()
    phi = t_st[2]
    el1 = dt*(v_c*np.cos(phi)-(v_c/L)*np.tan(alpha)*(a*np.sin(phi)+b*np.cos(phi)))
    el2 = dt*(v_c*np.sin(phi)+(v_c/L)*np.tan(alpha)*(a*np.cos(phi)-b*np.sin(phi)))
    el3 = dt*(v_c/L)*np.tan(alpha)
    el31 = slam_utils.clamp_angle(el3)
    motion = np.array([[el1],[el2],[el31]])
    el13 = -dt*v_c*(np.sin(phi)+(1/L)*np.tan(alpha)*(a*np.cos(phi)-b*np.sin(phi)))
    el23 = dt*v_c*(np.cos(phi)-(1/L)*np.tan(alpha)*(a*np.sin(phi)+b*np.cos(phi)))
    G = np.array([[1,0,el13],[0,1,el23],[0,0,1]])

    return motion, G

def odom_predict(u, dt, ekf_state, vehicle_params, sigmas, step):
    '''
    Perform the propagation step of the EKF filter given an odometry measurement u
    and time step dt where u = (ve, alpha) as shown in the vehicle/motion model.

    Returns the new ekf_state.
    '''
    t_st = ekf_state['x'].copy()
    t_st = np.reshape(t_st, (t_st.shape[0], 1))
    t_cov = ekf_state['P'].copy()
    dim = t_st.shape[0] - 3
    F_x = np.hstack((np.eye(3), np.zeros((3, dim))))
    mot, g = motion_model(u, dt, ekf_state, vehicle_params)
    new_x = t_st + np.matmul(np.transpose(F_x), mot)

    R_t = np.zeros((3, 3))
    R_t[0, 0] = sigmas['xy'] * sigmas['xy']
    R_t[1, 1] = sigmas['xy'] * sigmas['xy']
    R_t[2, 2] = sigmas['phi'] * sigmas['phi']
    Gt_1 = np.hstack((g, np.zeros((3, dim))))
    Gt_2 = np.hstack((np.zeros((dim, 3)), np.eye(dim)))
    Gt = np.vstack((Gt_1, Gt_2))
    new_cov = np.matmul(Gt, np.matmul(t_cov, np.transpose(Gt))) + np.matmul(np.transpose(F_x), np.matmul(R_t, F_x))
    new_cov = slam_utils.make_symmetric(new_cov)
    new_x = np.reshape(new_x, (new_x.shape[0],))
    ekf_state['x'] = new_x
    ekf_state['P'] = new_cov

    return ekf_state


def gps_update(gps, ekf_state, sigmas):
    '''
    Perform a measurement update of the EKF state given a GPS measurement (x,y).

    Returns the updated ekf_state.
    '''
    P = ekf_state['P']
    dim  = P.shape[0]-2
    H = np.hstack((np.eye(2),np.zeros((2,dim))))
    r = np.transpose([gps - ekf_state['x'][:2]])
    Q = (sigmas['gps']**2)*(np.eye(2))
    S = np.matmul(np.matmul(H,P),H.T) + Q
    S_inv = slam_utils.invert_2x2_matrix(S)
    d = np.matmul(np.matmul(r.T,S_inv),r)
    if d <= chi2.ppf(0.999, 2):
        K = np.matmul(np.matmul(P,H.T),S_inv)
        ekf_state['x'] = ekf_state['x'] + np.squeeze(np.matmul(K,r))
        ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
        P_temp = np.matmul((np.eye(P.shape[0])- np.matmul(K,H)),P)
        ekf_state['P'] = slam_utils.make_symmetric(P_temp)
        #print(f"[GPS Update] Covariance matrix after GPS measurement:\n{ekf_state['P']}")


    return ekf_state

def laser_measurement_model(ekf_state, landmark_id):
    '''
    Returns the measurement model for a (range,bearing) sensor observing the
    mapped landmark with id 'landmark_id' along with its jacobian.

    Returns:
        h(x, l_id): the 2x1 predicted measurement vector [r_hat, theta_hat].

        dh/dX: For a measurement state with m mapped landmarks, i.e. a state vector of
                dimension 3 + 2*m, this should return the full 2 by 3+2m Jacobian
                matrix corresponding to a measurement of the landmark_id'th feature.
    '''
    t_st = ekf_state['x'].copy()
    t_st[2] = slam_utils.clamp_angle(t_st[2])
    dim = t_st.shape[0]
    r_x,r_y,phi = t_st[0],t_st[1],t_st[2]
    m_x,m_y = t_st[3+2*landmark_id],t_st[4+2*landmark_id]
    del_x = m_x - r_x
    del_y = m_y - r_y
    q = (del_x)**2+(del_y)**2
    sqrt_q = np.sqrt(q)
    zhat = [[sqrt_q],[slam_utils.clamp_angle(np.arctan2(del_y,del_x)-phi)]]
    h = np.array([[-sqrt_q*del_x,-sqrt_q*del_y,0,sqrt_q*del_x,sqrt_q*del_y],[del_y,-del_x,-q,-del_y,del_x]])/q
    F_x = np.zeros((5,dim))
    F_x[:3,:3] = np.eye(3)
    F_x[3,3+2*landmark_id]=1
    F_x[4,4+2*landmark_id]=1
    H = np.matmul(h,F_x)

    return zhat, H

def initialize_landmark(ekf_state, tree):
    """
    Initialize a newly observed landmark in the filter state, increasing its
    dimension by 2.

    Returns the new ekf_state.
    """
    t_x, t_y, phi = ekf_state['x'][0], ekf_state['x'][1], ekf_state['x'][2]
    m_r, m_th = tree[0], tree[1]

    m_x = t_x + m_r * np.cos(m_th + phi)
    m_y = t_y + m_r * np.sin(m_th + phi)
    ekf_state['x'] = np.hstack((ekf_state['x'], m_x, m_y))

    # xpand the covariance matrix to include the new landmark
    temp_p = ekf_state['P']
    temp_p = np.vstack((temp_p, np.zeros((2, temp_p.shape[1]))))
    temp_p = np.hstack((temp_p, np.zeros((temp_p.shape[0], 2))))
    temp_p[-2, -2] = 1000  # init uncertainty for the new landmark
    temp_p[-1, -1] = 1000  
    ekf_state['P'] = temp_p
    ekf_state['num_landmarks'] += 1

    ekf_state['landmark_ages'].append(0)  #new landmark starts with age 0
    ekf_state['observation_count'].append(1)  

    return ekf_state


def dynamic_landmark_selection(ekf_state, num_selected, sigmas):
    """
    Selects a subset of landmarks based on Mahalanobis distance or covariance trace.
    Returns the indices of the selected landmarks.
    """
    num_landmarks = ekf_state['num_landmarks']
    if num_landmarks == 0:
        return []

    # compute Mahalanobis d
    mahalanobis_distances = []
    for i in range(num_landmarks):
        _, H = laser_measurement_model(ekf_state, i)
        Q_t = np.array([[sigmas['range']**2, 0], [0, sigmas['bearing']**2]])
        S = np.matmul(H, np.matmul(ekf_state['P'], H.T)) + Q_t
        distance = np.trace(S)  
        mahalanobis_distances.append((i, distance))

    # sort landmarks by distance or uncertainty
    mahalanobis_distances.sort(key=lambda x: x[1])
    selected_indices = [idx for idx, _ in mahalanobis_distances[:num_selected]]

    return selected_indices

def prune_landmarks(ekf_state, max_covariance_threshold, redundancy_threshold, min_observations, grace_period):
   
    if 'landmark_ages' not in ekf_state:
        ekf_state['landmark_ages'] = [0] * ekf_state['num_landmarks']

    if len(ekf_state['landmark_ages']) != ekf_state['num_landmarks']:
        ekf_state['landmark_ages'] = ekf_state['landmark_ages'][:ekf_state['num_landmarks']]

    if len(ekf_state['observation_count']) != ekf_state['num_landmarks']:
        ekf_state['observation_count'] = ekf_state['observation_count'][:ekf_state['num_landmarks']]

    num_landmarks = ekf_state['num_landmarks']
    if num_landmarks == 0:
        return ekf_state

    landmarks = ekf_state['x'][3:].reshape(-1, 2)  
    prune_indices = []
    redundant_indices = []
    grace_period_mask = np.array(ekf_state.get('landmark_ages', [0] * num_landmarks)) >= grace_period

    #covariance trace pruning
    for i in range(num_landmarks):
        P_i = ekf_state['P'][3 + 2 * i:3 + 2 * i + 2, 3 + 2 * i:3 + 2 * i + 2]
        trace_value = np.trace(P_i)
        if trace_value > max_covariance_threshold and grace_period_mask[i]:
            prune_indices.append(i)

    #redundancy pruning
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            dist = np.linalg.norm(landmarks[i] - landmarks[j])
            if dist < redundancy_threshold:
                redundant_indices.append(j)

    #observation-based pruning
    low_observation_indices = [
        i for i, count in enumerate(ekf_state.get('observation_count', [0] * num_landmarks))
        if count < min_observations and grace_period_mask[i]
    ]

    prune_indices = list(set(prune_indices + redundant_indices + low_observation_indices))

    for index in sorted(prune_indices, reverse=True):
        ekf_state['x'] = np.delete(ekf_state['x'], [3 + 2 * index, 3 + 2 * index + 1])
        ekf_state['P'] = np.delete(ekf_state['P'], [3 + 2 * index, 3 + 2 * index + 1], axis=0)
        ekf_state['P'] = np.delete(ekf_state['P'], [3 + 2 * index, 3 + 2 * index + 1], axis=1)
        ekf_state['landmark_ages'].pop(index)
        ekf_state['observation_count'].pop(index)
        ekf_state['num_landmarks'] -= 1

    ekf_state['landmark_ages'] = ekf_state['landmark_ages'][:ekf_state['num_landmarks']]
    ekf_state['observation_count'] = ekf_state['observation_count'][:ekf_state['num_landmarks']]
    return ekf_state

def laser_update(trees, assoc, ekf_state, sigmas, filter_params, max_landmarks=10):
    """
    Perform a measurement update of the EKF state given a set of tree measurements.
    Updates landmarks dynamically and prunes unreliable ones.
    """
    if filter_params is None:
        raise ValueError("filter_params cannot be None")

    
    if 'observation_count' not in ekf_state:
        ekf_state['observation_count'] = [0] * ekf_state['num_landmarks']

    # prune landmarks
    ekf_state = prune_landmarks(
      ekf_state,
      max_covariance_threshold=filter_params["max_covariance_threshold"],
      redundancy_threshold=1.0,
      min_observations=2,           
      grace_period=filter_params.get("grace_period", 10)                  
    )
    # validate consistency after pruning
    if len(ekf_state['landmark_ages']) != ekf_state['num_landmarks']:
        ekf_state['landmark_ages'] = ekf_state['landmark_ages'][:ekf_state['num_landmarks']]
        
    if len(ekf_state['observation_count']) != ekf_state['num_landmarks']:
        ekf_state['observation_count'] = ekf_state['observation_count'][:ekf_state['num_landmarks']]


    # select top-N landmarks dynamically
    selected_landmarks = dynamic_landmark_selection(ekf_state, max_landmarks, sigmas)

    Q_t = np.array([[sigmas['range']**2, 0], [0, sigmas['bearing']**2]])

    for i in range(len(trees)):
        j = assoc[i]

        if j == -1:
            ekf_state = initialize_landmark(ekf_state, trees[i])
            ekf_state['observation_count'].append(1)  # starts tracking observations
        elif j == -2 or j not in selected_landmarks:
            # discard unselected landmarks
            continue
        else:
            ekf_state['observation_count'][j] += 1

            #perform EKF 
            dim = ekf_state['x'].shape[0]
            z_hat, H = laser_measurement_model(ekf_state, j)

            S = np.matmul(H, np.matmul(ekf_state['P'], H.T)) + Q_t
            S_inv = np.linalg.inv(S)
            K = np.matmul(np.matmul(ekf_state['P'], H.T), S_inv)
            z = np.zeros((2, 1))
            z[0, 0] = trees[i][0]
            z[1, 0] = trees[i][1]

            inno = z - z_hat
            temp_st = ekf_state['x'] + np.squeeze(np.matmul(K, inno))
            temp_st[2] = slam_utils.clamp_angle(temp_st[2])
            ekf_state['x'] = temp_st
            temp_p = np.matmul((np.eye(dim) - np.matmul(K, H)), ekf_state['P'])
            temp_p = slam_utils.make_symmetric(temp_p)
            ekf_state['P'] = temp_p

    return ekf_state



def compute_data_association(ekf_state, measurements, sigmas, params):
    '''
    Computes measurement data association.

    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from
    measurements to landmarks.

    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    '''

    if ekf_state["num_landmarks"] == 0:
        return [-1 for m in measurements]

    n_lmark = ekf_state['num_landmarks']
    n_scans = len(measurements)
    M = np.zeros((n_scans, n_lmark))
    Q_t = np.array([[sigmas['range']**2, 0], [0, sigmas['bearing']**2]])

    alpha = chi2.ppf(0.95, 2)
    beta = chi2.ppf(0.99, 2)
    A = alpha*np.ones((n_scans, n_scans))

    for i in range(n_lmark):
        zhat, H = laser_measurement_model(ekf_state, i)
        S = np.matmul(H, np.matmul(ekf_state['P'],H.T)) + Q_t
        Sinv = slam_utils.invert_2x2_matrix(S)
        for j in range(n_scans):
            temp_z = measurements[j][:2]
            res = temp_z - np.squeeze(zhat)
            M[j, i] = np.matmul(res.T, np.matmul(Sinv, res))

    M_new = np.hstack((M, A))
    pairs = slam_utils.solve_cost_matrix_heuristic(M_new)
    pairs.sort()

    pairs = list(map(lambda x:(x[0],-1) if x[1]>=n_lmark else (x[0],x[1]),pairs))
    assoc = list(map(lambda x:x[1],pairs))

    for i in range(len(assoc)):
        if assoc[i] == -1:
            for j in range(M.shape[1]):
                if M[i, j] < beta:
                    assoc[i] = -2
                    break

    return assoc


def run_ekf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas, prune_interval=100):
    last_odom_t = -1
    ekf_state = {
        'x': ekf_state_0['x'].copy(),
        'P': ekf_state_0['P'].copy(),
        'num_landmarks': ekf_state_0['num_landmarks'],
        'landmark_ages': ekf_state_0.get('landmark_ages', []),
        'observation_count': ekf_state_0.get('observation_count', [])
    }

    state_history = {
        't': [0],
        'x': ekf_state['x'],
        'P': np.diag(ekf_state['P'])
    }

    trajectory = []  
    if filter_params["do_plot"]:
        plot = slam_utils.init_plot()

    for i, event in enumerate(events):
        # Increment ages
        if 'landmark_ages' not in ekf_state:
            ekf_state['landmark_ages'] = [0] * ekf_state['num_landmarks']
        ekf_state['landmark_ages'] = [age + 1 for age in ekf_state['landmark_ages']]

        t = event[1][0]
        if i % 500 == 0:
            print(f"[Step {i}] Processing event: {event[0]} at time {t:.3f}")

        if i % prune_interval == 0:
            print(f"[Step {i}] Total landmarks before pruning: {ekf_state['num_landmarks']}")
            num_landmarks_before = ekf_state['num_landmarks']

            prune_landmarks(
                ekf_state,
                max_covariance_threshold=filter_params["max_covariance_threshold"],
                redundancy_threshold=filter_params.get("redundancy_threshold", 5.0),
                min_observations=filter_params.get("min_observations", 3),
                grace_period=filter_params.get("grace_period", 10)  # Add this line
            )


            num_landmarks_after = ekf_state['num_landmarks']
            pruned_landmarks = num_landmarks_before - num_landmarks_after
            print(f"[Step {i}] Pruned {pruned_landmarks} landmarks.")

        if event[0] == 'gps':
            gps_msmt = event[1][1:]
            ekf_state = gps_update(gps_msmt, ekf_state, sigmas)

        elif event[0] == 'odo':
            if last_odom_t < 0:
                last_odom_t = t
                continue
            u = event[1][1:]
            dt = t - last_odom_t
            ekf_state = odom_predict(u, dt, ekf_state, vehicle_params, sigmas, step=i)
            last_odom_t = t

        else:  
            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)
            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            if filter_params["do_plot"]:
                slam_utils.do_plot(state_history['x'], ekf_state, trees, scan, assoc, plot, filter_params)

        
        trajectory.append(ekf_state['x'][:3].copy()) 
        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:3]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:3, :3])))
        state_history['t'].append(t)

    return state_history, np.array(trajectory)  


def compute_rmse(estimated, gps_data):

    min_length = min(estimated.shape[0], gps_data.shape[0])
    estimated = estimated[:min_length, :2]  
    gps_data = gps_data[:min_length, :2]
    errors = estimated - gps_data
    squared_errors = np.square(errors)
    rmse_x = np.sqrt(np.mean(squared_errors[:, 0]))
    rmse_y = np.sqrt(np.mean(squared_errors[:, 1]))
    overall_rmse = np.sqrt(np.mean(np.sum(squared_errors, axis=1)))

    print(f"RMSE for x: {rmse_x:.4f} m")
    print(f"RMSE for y: {rmse_y:.4f} m")
    print(f"Overall RMSE: {overall_rmse:.4f} m")

def main():
    odo = slam_utils.read_data_file("vp_data/DRS.txt")
    gps = slam_utils.read_data_file("vp_data/GPS.txt")
    laser = slam_utils.read_data_file("vp_data/LASER.txt")

    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in laser])
    events.extend([('odo', x) for x in odo])

    events = sorted(events, key=lambda event: event[1][0])

    vehicle_params = {
        "a": 3.78,
        "b": 0.50,
        "L": 2.83,
        "H": 0.76
    }

    filter_params = {
      "max_laser_range": 75,  # meters
      "do_plot": True,
      "plot_raw_laser": False,
      "plot_map_covariances": False,
      "max_covariance_threshold": 30,  # Threshold for pruning based on covariance
      "redundancy_threshold": 2.0,     # Minimum distance between landmarks
      "min_observations": 2,           # Minimum observations to retain a landmark
      "grace_period": 15             # Grace period for newly added landmarks
    }


    sigmas = {
        "xy": 0.05,
        "phi": 0.5 * np.pi / 180,
        "gps": 3,
        "range": 0.5,
        "bearing": 5 * np.pi / 180
    }

    ekf_state = {
        "x": np.array([gps[0, 1], gps[0, 2], 36 * np.pi / 180]),
        "P": np.diag([0.1, 0.1, 1]),
        "num_landmarks": 0,
        "landmark_ages": [],
        "observation_count": []
    }

    _, trajectory = run_ekf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)
    compute_rmse (trajectory, gps)

if __name__ == '__main__':
    main()
