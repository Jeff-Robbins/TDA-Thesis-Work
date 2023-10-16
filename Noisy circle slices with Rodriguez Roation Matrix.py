import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import time
import random

from scipy.spatial.transform import Rotation as R

from tqdm import tqdm


# import pandas
import pprint

import pandas as pd

# from tqdm import tqdm
# import multiprocessing as mp

from joblib import Parallel, delayed

# from mpl_toolkits.mplot3d import Axes3D
from persim import plot_diagrams

# import matplotlib.gridspec as gridspec
# import networkx as nx
# from IPython.display import Video
from ripser import ripser
# import persim
# import teaspoon.MakeData.PointCloud as makePtCloud
# import teaspoon.TDA.Draw as Draw
# from teaspoon.SP.network import ordinal_partition_graph
# from teaspoon.TDA.PHN import PH_network
# from teaspoon.SP.network_tools import make_network
# from teaspoon.parameter_selection.MsPE import MsPE_tau
# import teaspoon.MakeData.DynSysLib.DynSysLib as DSL



# import ripser
# import persim

# from teaspoon.SP.network import ordinal_partition_graph
# from teaspoon.TDA.PHN import PH_network
# from teaspoon.SP.network_tools import make_network
# from teaspoon.parameter_selection.MsPE import MsPE_tau





# from IPython.display import Video

# scikit-tda imports..... Install all with -> pip install scikit-tda
#--- this is the main persistence computation workhorse
# import ripser
# from persim import plot_diagrams
# import persim
# import persim.plot

# teaspoon imports...... Install with -> pip install teaspoon
# these are for generating data and some drawing tools

# these are for generating time series network examples
# from teaspoon.SP.network import ordinal_partition_graph
# from teaspoon.TDA.PHN import PH_network
# from teaspoon.SP.network_tools import make_network
# from teaspoon.parameter_selection.MsPE import MsPE_tau


# sample_list = list(range(100, 501, 100))
sample_list = [1000]
dictionary = {key: [] for key in sample_list}
print( [random.randint(500,2000) for _ in range(10)] )

def fibonacci_sphere(samples):
    indices = np.arange(samples, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / samples)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi) # Each row vectors

    return np.column_stack((x, y, z)) # 3 x 1000 matrix











# Display point cloud
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

# Print point cloud data set
# print(points)
#
# type(points)



#normal = np.array([0.0, 0.0, 1.0]) # Z-axis normal
#normal = np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0]) # Normal to the plane x + y = 0
#normal /= np.linalg.norm(normal) # Normalizing vector


def get_rotation_matrix_to_z_axis(vec):
    # Normalize the input vector

    vec = vec / np.linalg.norm(vec)

    # Define the z-axis

    z_axis = np.array([0, 0, 1])

    # Calculate the angle between the vector and the z-axis

    theta = np.arccos(np.dot(vec, z_axis))

    # Calculate the axis of rotation

    axis_of_rotation = np.cross(vec, z_axis)

    axis_of_rotation = axis_of_rotation / np.linalg.norm(axis_of_rotation)

    # Construct the rotation matrix using the scipy's Rotation

    rotation_vector = theta * axis_of_rotation

    rotation = R.from_rotvec(rotation_vector)

    # Return the rotation matrix

    return rotation.as_matrix()



# Define a function that takes in a random normal vector, normalizes it and rotates space so that it lines up with the
#       z-axis, and its normal plane becomes the xy-plane
# def North_pole_normalizer(N,point_cloud_matrix):
#     Unit_N = abs(N)/np.linalg.norm(N)
#     M = np.array([[Unit_N[0],Unit_N[1],0]]) # Projection of Normal vector onto xy-plane
#     theta = np.arccos(np.dot(M,[1, 0, 0])/np.linalg.norm(M)) # Angle from x-axis
#     phi = np.arccos(np.dot(Unit_N,[0, 0, 1])/np.linalg.norm(Unit_N)) # Angle from z-axis
#     Ry = np.array([[np.cos(-phi),0,np.sin(-phi)],
#                    [0,1,0],
#                    [-np.sin(-phi),0,np.cos(-phi)]], dtype=float) # Rotational matrix about y-axis, 3x3 matrix
#     Rz = np.array([[1,0,0],
#                    [0,np.cos(-theta),-np.sin(-theta)],
#                    [0,np.sin(-theta),np.cos(-theta)]], dtype=float) # Rotational matrix about the z axis, 3x3 matrix
#     # Vert_N = Ry.dot(Rz).dot(Unit_N) # Should be the normal matrix rotated to the z-axis
#     Vert_N = Ry @ Rz @ Unit_N
#     # Transpose the matrix of points from 1000x3 to 3x1000
#     # Rot_pt_cloud = Ry.dot(Rz).dot(np.transpose(points)) # Rotating the point cloud to follow the normal
#     Rot_pt_cloud = np.transpose(Ry @ Rz @ np.transpose(points))  # Rotating the point cloud to follow the normal
#     return Vert_N, Rz, Ry, Unit_N, Rot_pt_cloud, M



#



#Vert_N, Rz, Ry, Unit_N, Rot_pt_cloud, M = North_pole_normalizer(normal, points)

#Vert_N = np.array([0,0,1])

#diagrams = ripser.ripser(Rot_pt_cloud )['dgms']
#drawTDAtutorial(Rot_pt_cloud, diagrams)

def get_fund_lifetime_of_S1_projected_data(n):
    points = fibonacci_sphere(n)

    # Apply radial noise
    mean = 1
    std_dev = 0
    noise = np.random.normal(mean, std_dev, n)
    points *= noise[:, np.newaxis]  # Scale radius by noise

    # Generate a random plane passing through the origin
    normal = np.random.randn(3)

    Rot_matrix = get_rotation_matrix_to_z_axis(normal)

    Rot_pt_cloud = Rot_matrix @ np.transpose(points)

    Rot_pt_cloud = np.transpose(Rot_pt_cloud)

    z_axis = np.array([0, 0, 1])

    # Project points within a distance of 0.1 from the plane
    distances = np.abs(np.dot(Rot_pt_cloud, z_axis))
    # distances = np.abs(np.dot(Rot_pt_cloud, xy_plane))
    mask = distances <= 0.1

    projected_points = Rot_pt_cloud[mask] - np.dot(Rot_pt_cloud[mask], np.outer(z_axis, z_axis))

    diagrams = ripser(projected_points, maxdim=1)['dgms']
    lifetime = max(abs(np.diff(diagrams[-1], axis=1)))[0]

    Sample_size = len(projected_points)

    return lifetime, Sample_size

# lifetime, Sample_size = get_fund_lifetime_of_S1_projected_data(1000)
#
# print(lifetime)
# print(Sample_size)



n_lifetime_df = np.zeros((10000, 2))


T1 = time.time()

for i in tqdm(range(10)):
    num_points = random.randint(500, 3000)
    print(num_points)
    for j in range(1000):
        lifetime, Sample_size = get_fund_lifetime_of_S1_projected_data(num_points)
        n_lifetime_df[j + i*1000, 0] = Sample_size
        n_lifetime_df[j + i*1000, 1] = lifetime


T2 = time.time()

print("This script took this long, in seconds: ", T2 - T1)

print(n_lifetime_df)

# projected_points = Rot_matrix @ np.transpose(projected_points)
#

#projected_points = Rot_pt_cloud[mask] - np.dot(Rot_pt_cloud[mask],np.outer(xy_plane, xy_plane))


# projected_points = points[mask] - np.outer(np.dot(points[mask], normal), normal)
# projected_points = Rot_pt_cloud[mask] - np.outer(np.dot(Rot_pt_cloud[mask], Vert_N), Vert_N)
# projected_points = points[mask] - np.dot(points[mask],np.outer(normal, normal))





# # Display point cloud
# fig = plt.figure()
# ax = fig.add_subplot(121, projection='3d')
# #ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5)
# ax.scatter(Rot_pt_cloud[:, 0], Rot_pt_cloud[:, 1], Rot_pt_cloud[:, 2], s=5)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Point Cloud')
#
# # Display projected points on the plane
# ax2 = fig.add_subplot(122)
# ax2.scatter(projected_points[:, 0], projected_points[:, 1], s=5)
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_title('Projected Points')
#
# # Set the aspect ratio of the plot to 'equal' for orthogonal perspective
# ax2.set_aspect('equal')
#
# plt.show()
#
#
# # Step 2: Compute persistent homology
# dgms2 = ripser(Rot_pt_cloud,maxdim=2)['dgms']
#
# fig = plt.figure(figsize=(9, 3))
# ax = fig.add_subplot(131, projection='3d')
# ax.scatter(Rot_pt_cloud[:, 0], Rot_pt_cloud[:, 1], Rot_pt_cloud[:, 2])
# ax.set_aspect('equal')
# plt.title("Generated Loop")
# plt.subplot(132)
# plot_diagrams(dgms2)
# plt.title("$\mathbb{Z} / 2$")
#
# plt.tight_layout()
# plt.show()






# import numpy as np
# import matplotlib.pyplot as plt
# from ripser import ripser
# # from ripser.plot import plot_diagrams
#
#
# diagrams = ripser(Rot_pt_cloud, maxdim=2)['dgms']
#
# plot_diagrams(diagrams, show=True)


# from ripser import Rips
# from persim import plot_diagrams
#
# rips = Rips(maxdim = 2)
#
# dgms = rips.fit_transform(Rot_pt_cloud)
# # persim.plot_diagrams(dgms)
# plot_diagrams(dgms)
#
# plt.show(dgms)



# import numpy as np
# from ripser import ripser
# import matplotlib.pyplot as plt
# from persim import plot_diagrams
#
#
#
# # Compute the persistence diagrams
# diagrams = ripser(Rot_pt_cloud, maxdim=2)['dgms']
#
# # Plot the persistence diagrams
# plot_diagrams(diagrams)
# plt.show()


# import numpy as np
# from ripser import ripser
# from persim import plot_diagrams
#
# # Generate a random 3-dimensional point cloud
# point_cloud = np.random.rand(100, 3)
#
# # Compute the persistence diagrams
# diagrams = ripser(Rot_pt_cloud, maxdim=2)['dgms']
#
# # Plot the persistence diagrams
# plot_diagrams(diagrams)
# plt.show()
#
#
# # North Hemisphere
# N_hemi = Rot_pt_cloud[Rot_pt_cloud[:2] > 0]




# Lifetimes_list = []
#
#
# def method(num_points,my_dict):
#     with Parallel(n_jobs=-1,backend = "multiprocessing") as parallel:
#         lifetimes = parallel(delayed(get_fund_lifetime_of_S1_projected_data(num_points)) for _ in range(10))
#     my_dict[num_points].extend(lifetimes)
#     return my_dict
#
#     # for j in range(10):
#     #     with Parallel(n_jobs=10) as parallel:
#     #         parallel(delayed(get_fund_lifetime_from_pt_cloud)(num_points, dictionary) for num_points in sample_list)
#     #     lifetime = get_fund_lifetime_from_pt_cloud(num_points)
#     #     # n_lifetime_df[j + 50 * i, 0] = num_points
#     #     # n_lifetime_df[j + 50 * i, 1] = lifetime
#     #     my_dict[num_points].append(lifetime)
#     #     print(lifetime)
#     # return Lifetimes_list
#
# # output = method(300,dictionary)
#
# T1 = time.time()
#
# # with Parallel(n_jobs=10, backend="threading") as parallel:
# #     parallel(delayed(method)(num_points, dictionary) for num_points in sample_list)
#
# # n_lifetime_df = np.zeros((1000, 2))
#
# with Parallel(n_jobs=5) as parallel: # Using 10 cores
#     results = parallel(delayed(method)(num_points, dictionary) for num_points in sample_list)
#
# for result in results:
#     for key, value in result.items():
#         dictionary[key].extend(value)