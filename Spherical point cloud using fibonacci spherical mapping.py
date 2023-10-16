import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D





import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from IPython.display import Video

# scikit-tda imports..... Install all with -> pip install scikit-tda
#--- this is the main persistence computation workhorse
import ripser
# from persim import plot_diagrams
import persim
# import persim.plot

# teaspoon imports...... Install with -> pip install teaspoon
#---these are for generating data and some drawing tools
import teaspoon.MakeData.PointCloud as makePtCloud
import teaspoon.TDA.Draw as Draw

#---these are for generating time series network examples
from teaspoon.SP.network import ordinal_partition_graph
from teaspoon.TDA.PHN import PH_network
from teaspoon.SP.network_tools import make_network
from teaspoon.parameter_selection.MsPE import MsPE_tau
import teaspoon.MakeData.DynSysLib.DynSysLib as DSL



def fibonacci_sphere(samples):
    indices = np.arange(samples, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / samples)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi) # Each row vectors

    return np.column_stack((x, y, z)) # 3 x 1000 matrix

# Generate point cloud
num_points = 1000
points = fibonacci_sphere(num_points) # 1000 x 3 matrix


# Apply radial noise
mean = 1
std_dev = 0.05
noise = np.random.normal(mean, std_dev, num_points)
points *= noise[:, np.newaxis]  # Scale radius by noise


# def drawTDAtutorial(P, diagrams, R=3):
#     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
#
#     # Draw point cloud
#     plt.sca(axes[0])
#     plt.title('Point Cloud')
#     plt.scatter(P[:, 0], P[:, 1])
#
#     # Draw diagrams
#     plt.sca(axes[1])
#     plt.title('0-dim Diagram')
#     Draw.drawDgm(diagrams[0])
#
#     plt.sca(axes[2])
#     plt.title('1-dim Diagram')
#     Draw.drawDgm(diagrams[1])
#     plt.axis([0, R, 0, R])

# diagrams = ripser.ripser(points)['dgms']
# drawTDAtutorial(points,diagrams)


# Display point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Print point cloud data set
# print(points)
#
# type(points)

# Generate a random plane passing through the origin
normal = np.random.randn(3)
# normal = np.array([0.0, 0.0, 1.0]) # Z-axis normal
# normal = np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0]) # Normal to the plane x + y = 0
normal /= np.linalg.norm(normal) # Normalizing vector



# Define a function that takes in a random normal vector, normalizes it and rotates space so that it lines up with the
#       z-axis, and its normal plane becomes the xy-plane
# def North_pole_normalizer(N,point_cloud_matrix):
#     Unit_N = abs(N)/np.linalg.norm(N)
#     M = np.array([[Unit_N[0],Unit_N[1],0]]) # Projection of Normal vector onto xy-plane
#     theta = np.arccos(np.dot(M,[1, 0, 0])/np.linalg.norm(M)) # Angle from x-axis
#     phi = np.arccos(np.dot(Unit_N,[0, 0, 1])/np.linalg.norm(Unit_N)) # Angle from z-axis
#     Ry = np.array([[np.cos(-phi),0,np.sin(-phi)],
#                    [0,1,0],
#                    [-np.sin(-phi),0,np.cos(-phi)]],dtype=object) # Rotational matrix about y-axis, 3x3 matrix
#     Rz = np.array([[1,0,0],
#                    [0,np.cos(-theta),-np.sin(-theta)],
#                    [0,np.sin(-theta),np.cos(-theta)]], dtype=object) # Rotational matrix about the z axis, 3x3 matrix
#     Vert_N = Ry.dot(Rz).dot(Unit_N) # Should be the normal matrix rotated to the z-axis
#     # Transpose the matrix of points from 1000x3 to 3x1000
#     Rot_pt_cloud = Ry.dot(Rz).dot(np.transpose(points)) # Rotating the point cloud to follow the normal
#     Trans_points = np.transpose(Rot_pt_cloud)
#     return Vert_N, Trans_points
#
# Vert_N, Rot_pt_cloud = North_pole_normalizer(normal, points)
# Project points within a distance of 0.1 from the plane
distances = np.abs(np.dot(points, normal))
# distances = np.abs(np.dot(Rot_pt_cloud, Vert_N))
mask = distances <= 0.1
projected_points = points[mask] - np.outer(np.dot(points[mask], normal), normal)
# projected_points = Rot_pt_cloud[mask] - np.outer(np.dot(Rot_pt_cloud[mask], Vert_N), Vert_N)


# projected_points = points[mask] - np.dot(points[mask],np.outer(normal, normal))
# projected_points = Rot_pt_cloud[mask] - np.dot(Rot_pt_cloud[mask],np.outer(Vert_N, Vert_N))

# Display point cloud
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5)
# ax.scatter(Rot_pt_cloud[:, 0], Rot_pt_cloud[:, 1], Rot_pt_cloud[:, 2], s=5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud')

# Display projected points on the plane
ax2 = fig.add_subplot(122)
ax2.scatter(projected_points[:, 0], projected_points[:, 1], s=5)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Projected Points')

# Set the aspect ratio of the plot to 'equal' for orthogonal perspective
ax2.set_aspect('equal')

plt.show()

