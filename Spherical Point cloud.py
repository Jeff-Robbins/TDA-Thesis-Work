import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_point_cloud(num_points, radius):
    points = []
    for _ in range(num_points):
        theta = np.random.uniform(0,2*np.pi)
        phi = np.arccos(2 * np.random.uniform(0,1) - 1)

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        points.append((x, y, z))

    return points

# Generate a point cloud with 1000 points on the surface of a sphere with radius 1
point_cloud = generate_point_cloud(1000, 1)

# Convert point cloud to numpy array for easier manupulation
point_cloud = np.array(point_cloud)

# Display the point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(point_cloud[:, 0], point_cloud[:, 2], c='b', marker='.')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Enable interactive rotation
def on_mouse_move(event):
    if event.inaxes == ax:
        ax.view_init(elev=10, azim=event.xdata)

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)


plt.show()