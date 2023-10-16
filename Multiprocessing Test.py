import numpy as np
# import matplotlib.pyplot as plt
# import pandas
import pprint

import pandas as pd
from ripser import ripser
# from tqdm import tqdm
# import multiprocessing as mp

from joblib import Parallel, delayed

import time

# import multiprocessing
# from persim import plot_diagrams


# def fibonacci_sphere(samples):
#     indices = np.arange(samples, dtype=float) + 0.5
#     phi = np.arccos(1 - 2 * indices / samples)
#     theta = np.pi * (1 + 5 ** 0.5) * indices
#
#     x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi) # Each row vectors
#
#     return np.column_stack((x, y, z)) # 3 x 1000 matrix
# # Generate point cloud
# # num_points = 1000
#
#  # 1000 x 3 matrix
# def apply_radial_noise(num_points):
#     points = fibonacci_sphere(num_points)
#     # Apply radial noise
#     mean = 1
#     std_dev = 0.05
#     noise = np.random.normal(mean, std_dev, num_points)
#     points *= noise[:, np.newaxis]  # Scale radius by noise
#     return points
# '''
# Function that takes in a pt_cloud and outputs the  lifetime of the fundamental class.
#        Have to ensure that all iterations only contain a single point in H2, otherwise find the max lifetime. Check
#        if the length of the array is greater than 1 (row) if it is, run a loop ranging over the length of the array
#        calculating the lifetime of each row in the array and place these values in an array. Add a column to this array
#        of integers from 0 to the length of the array so now you have an nx2 array. Order the array from greatest to
#        lowest lifetimes, then select the the row of the diagrams array that is the integer value in the 0'th row after
#        ordering the lifetime's array.
# '''
# def get_fund_lifetime(pt_cloud):
#     diagrams = ripser(pt_cloud, maxdim=2)['dgms']
#     lifetime = max(abs(np.diff(diagrams[-1], axis=1)))[0]
#     return lifetime

# sample_list = list(range(100, 501, 100))
sample_list = [1000]
dictionary = {key: [] for key in sample_list}

def get_fund_lifetime_from_pt_cloud(num_points):
    # Generate fibonacci spherical pt_cloud
    indices = np.arange(num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)  # Each row vectors
    points = np.column_stack((x, y, z))

    # Apply radial noise
    mean = 1
    std_dev = 0.05
    noise = np.random.normal(mean, std_dev, num_points)
    points *= noise[:, np.newaxis]  # Scale radius by noise

    # Get fund_lifetime
    diagrams = ripser(points, maxdim=2)['dgms']
    lifetime = max(abs(np.diff(diagrams[-1], axis=1)))[0]

    return lifetime

Lifetimes_list = []


def method(num_points,my_dict):
    with Parallel(n_jobs=-1, backend = "threading") as parallel:
        lifetimes = parallel(delayed(get_fund_lifetime_from_pt_cloud)(num_points) for _ in range(1))
    my_dict[num_points].extend(lifetimes)
    return my_dict

    # for j in range(10):
    #     with Parallel(n_jobs=10) as parallel:
    #         parallel(delayed(get_fund_lifetime_from_pt_cloud)(num_points, dictionary) for num_points in sample_list)
    #     lifetime = get_fund_lifetime_from_pt_cloud(num_points)
    #     # n_lifetime_df[j + 50 * i, 0] = num_points
    #     # n_lifetime_df[j + 50 * i, 1] = lifetime
    #     my_dict[num_points].append(lifetime)
    #     print(lifetime)
    # return Lifetimes_list

# output = method(300,dictionary)

T1 = time.time()

# with Parallel(n_jobs=10, backend="threading") as parallel:
#     parallel(delayed(method)(num_points, dictionary) for num_points in sample_list)

# n_lifetime_df = np.zeros((1000, 2))

with Parallel(n_jobs=5) as parallel: # Using 10 cores
    results = parallel(delayed(method)(num_points, dictionary) for num_points in sample_list)

for result in results:
    for key, value in result.items():
        dictionary[key].extend(value)


# for i in [100,200,300]:
#     num_points = i
#     output = method(i,dictionary)

# results = Parallel(n_jobs=10)(delayed(method)(num_points,dictionary) for num_points in sample_list)

T2 = time.time()

print(f"This script took {T2 - T1} seconds")

# pprint.pp(dictionary)
# print("Here is the dictionary ", dictionary)
# print(Lifetimes_list)



# print(n_lifetime_df)
#
# for i in tqdm(range(10)):
#     num_points = 100 + 100*i
#     for j in range(1):
#         pt_cloud = apply_radial_noise(num_points)
#         lifetime = get_fund_lifetime(pt_cloud)
#         n_lifetime_df[j+50*i, 0] = num_points
#         n_lifetime_df[j+50*i, 1] = lifetime

'''
 Initialize the array of 2 dimensions, the number of datapoints n, and the lifetime of the fundamental class for each
 iteration
'''

# processes = []

# for _ in range(10):
#     p = mp.Process(target=get_fund_lifetime_from_pt_cloud, args=[100])
#     p.start()
#     processes.append(p)
#     num_points, lifetime = get_fund_lifetime_from_pt_cloud(num_points)
#
# for process in processes:
#     process.join()
#
# for i in range(10):
#     if n_lifetime_df[i, 0] == 0:
#         n_lifetime_df[i, 0] = num_points
#     elif n_lifetime_df[i, 1] == 0:
#         n_lifetime_df[i, 1] = lifetime







# n_lifetime_df = pandas.DataFrame(data = n_lifetime_df, columns = ['Sample size', 'Lifetime of Fundamental Class'])
#
# print(n_lifetime_df)
#
#
# n_lifetime_df.boxplot(by='Sample size', column =['Lifetime of Fundamental Class'],
#                                 grid=False)
#
# plt.axhline(y = np.sqrt(3), color='r', linestyle = '-')
#
# plt.show()
