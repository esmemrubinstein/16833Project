
import numpy as np
import matplotlib.pyplot as plt

def read_data_file(file_name):
    with open(file_name, "r") as f:
        raw_data = f.readlines()

    data = [ [float(x) for x in line.strip().split(',')] for line in raw_data ]

    return np.array(data)


def tree_to_global_xy(trees, ekf_state):
    if len(trees) == 0:
        return []

    trees = np.array(trees) # rows are [range, bearing, diameter]
    phi = ekf_state["x"][2]
    mu = ekf_state["x"][0:2]

    return np.reshape(mu, (2,1)) + np.vstack(( trees[:,0]*np.cos(phi+trees[:,1]), 
                                                trees[:,0]*np.sin(phi+trees[:,1])))

def plot_tree_measurements(trees, assoc, ekf_state, plot):
    if len(trees) == 0:
        return

    G_trees = tree_to_global_xy(trees, ekf_state)
    mu = ekf_state["x"][0:2]

    t_list = [t.tolist() for t in G_trees.T]

    if "lasers" not in plot:
        plot["lasers"] = []

    for i in range(len(assoc)):
        data = np.vstack((mu, t_list[i]))

        if assoc[i] >= 0:
            color = 'g'
        elif assoc[i] == -2:
            color = 'r'
        else:
            color = 'b'

        if i >= len(plot["lasers"]):
            plot["lasers"].append(plot["axis"].plot(data[0], data[1], color=color, lw=2))
        else:
            plot["lasers"][i][0].set_data(data[0], data[1])
            plot["lasers"][i][0].set_color(color)

def plot_trajectory(traj, plot):
    if np.prod(traj.shape) > 3:
        if "trajectory" not in plot:
            plot["trajectory"] = plot["axis"].plot([], [], color='k')[0]

        plot["trajectory"].set_data(traj[:, 0], traj[:, 1])

def plot_map(ekf_state, plot, params):
    if "map" not in plot:
        plot["map"] = plot["axis"].plot([], [], 'g+', markersize=13)[0]

    lms = np.reshape(ekf_state["x"][3:], (-1, 2))
    plot["map"].set_data(lms[:, 0], lms[:, 1])

    if params["plot_map_covariances"]:
        if "map_covariances" not in plot:
            plot["map_covariances"] = []

        for i in range(ekf_state["num_landmarks"]):
            idx = 3 + 2 * i
            P = ekf_state["P"][idx:idx + 2, idx:idx + 2]

            circ = get_covariance_ellipse_points(ekf_state["x"][idx:idx + 2], P)

            if i >= len(plot["map_covariances"]):
                plot["map_covariances"].append(plot["axis"].plot([], [], 'b-')[0])

            plot["map_covariances"][i].set_data(circ[:, 0], circ[:, 1])
        

def get_covariance_ellipse_points(mu, P, base_circ=[]):

    if len(base_circ) == 0:
        N = 20
        phi = np.linspace(0, 2*np.pi, N)
        x = np.reshape(np.cos(phi), (-1,1))
        y = np.reshape(np.sin(phi), (-1,1))
        base_circ.extend(np.hstack((x,y)).tolist())

    vals, _ = np.linalg.eigh(P)

    offset = 1e-6 - min(0, vals.min())

    G = np.linalg.cholesky(P + offset * np.eye(mu.shape[0]))

    # 3 sigma bound
    circ = 3*np.matmul(np.array(base_circ), G.T) + mu

    return circ

def convert_to_global_xy(ekf_state, scan, params):
    angles = np.array(range(361))*np.pi/360 - np.pi/2

    rb = np.vstack((scan, angles)).T
    phi = ekf_state["x"][2]
    mu = ekf_state["x"][0:2]

    rb = rb[ rb[:,0] < params["max_laser_range"], : ]

    xy = np.reshape(mu, (2,1)) + np.vstack(( rb[:,0]*np.cos(phi+rb[:,1]), 
                                                rb[:,0]*np.sin(phi+rb[:,1])))

    return xy

def plot_scan(ekf_state, scan, plot, params):
    if "scan" not in plot:
        plot["scan"] = plot["axis"].plot([], [], 'kd', markersize=3)[0]

    scan = convert_to_global_xy(ekf_state, scan, params)

    plot["scan"].set_data(scan[:, 0], scan[:, 1])

def plot_robot(ekf_state, plot):
    # base triangle shape
    triangle = 1.5 * np.array([[0, 0], [-3, 1], [-3, -1], [0, 0]])

    # rotate to correct orientation
    phi = ekf_state["x"][2]
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    triangle = np.matmul(triangle, R.T)

    # translate to correct position
    triangle += ekf_state["x"][:2]

    if "robot" not in plot:
        plot["robot"] = plot["axis"].plot([], [], 'k-', lw=2)[0]

    plot["robot"].set_data(triangle[:, 0], triangle[:, 1])

def plot_covariance(ekf_state, plot):
    Pp = ekf_state["P"][:2, :2]

    circ = get_covariance_ellipse_points(ekf_state["x"][:2], Pp)

    if "cov" not in plot:
        plot["cov"] = plot["axis"].plot([], [], 'b-')[0]

    plot["cov"].set_data(circ[:, 0], circ[:, 1])

def plot_state(ekf_state, plot, params):
    plot_map(ekf_state, plot, params)
    plot_robot(ekf_state, plot)
    plot_covariance(ekf_state, plot)

def do_plot(xhist, ekf_state, trees, scan, assoc, plot, params):
    plot_trajectory(xhist, plot)
    plot_state(ekf_state, plot, params)

    plt.draw()
    plt.pause(0.1)
    

def init_plot():
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))  # Set a reasonable figure size for screen display

    # Set axis limits to be from 0 to 100 for both x and y axes
    ax.set_xlim(-250, 100)  # Set x-axis limit from 0 to 100
    ax.set_ylim(-200, 200)  # Set y-axis limit from 0 to 100

    ax.set_aspect('equal', 'box')  # Ensures the aspect ratio is maintained
    ax.set_title("EKF SLAM")

    plot = {
        "fig": fig,
        "axis": ax
    }

    return plot


def clamp_angle(theta):
    while theta >= np.pi:
        theta -= 2*np.pi

    while theta < -np.pi:
        theta += 2*np.pi

    return theta

def make_symmetric(P):
    return 0.5 * (P + P.T)

def invert_2x2_matrix(M):
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    return np.array([ [M[1,1], -M[0,1]], [-M[1,0], M[0,0]] ]) / det

def solve_cost_matrix_heuristic(M):
    n_msmts = M.shape[0]
    result = []

    ordering = np.argsort(M.min(axis=1))

    for msmt in ordering:
        match = np.argmin(M[msmt,:])
        M[:, match] = 1e8
        result.append((msmt, match))

    return result