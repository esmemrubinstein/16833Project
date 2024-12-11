import numpy as np


unmanaged_pose = 'results/pose_no_management.txt'
managed_pose = ''

unmanaged_pose = np.loadtxt(unmanaged_pose)
managed_pose = np.loadtxt(managed_pose)

unmanaged_x, unmanaged_y, unmanaged_theta = unmanaged_pose[:, 0], unmanaged_pose[:, 1], unmanaged_pose[:, 2]
managed_x, managed_y, managed_theta = managed_pose[:, 0], managed_pose[:, 1], managed_pose[:, 2]

euclidean_distance = np.sqrt((unmanaged_x - managed_x)**2 + (unmanaged_y - managed_y)**2)
avg_dist = np.mean(euclidean_distance)

x_diff_squared = (unmanaged_x - managed_x)**2
y_diff_squared = (unmanaged_y - managed_y)**2
theta_diff_squared = (unmanaged_theta - managed_theta)**2

rmse = np.sqrt(np.mean(x_diff_squared + y_diff_squared + theta_diff_squared))

print("Average Euclidean distance between managed and unmanaged poses: ", avg_dist)
print("Root Mean Square Error between managed and unmanaged poses: ", rmse)