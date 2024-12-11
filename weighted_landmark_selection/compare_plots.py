import numpy as np
import matplotlib.pyplot as plt

# File paths (replace with your actual file names)
managed = ''
unmanaged = 'results/pose_no_management.txt'

data1 = np.loadtxt(managed)
data2 = np.loadtxt(unmanaged)

x1, y1 = data1[:, 0], data1[:, 1]
x2, y2 = data2[:, 0], data2[:, 1]


plt.figure(figsize=(10, 6))
plt.plot(x1, y1, 'o', markersize=1, label='Managed', alpha=0.7)  # 'o' for scatter points
plt.plot(x2, y2, 'x', markersize=1, label='Unmanaged', alpha=0.7)  # 'x' for different marker

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Comparison of Robot Pose')
plt.legend()
plt.grid(True)

plt.show()