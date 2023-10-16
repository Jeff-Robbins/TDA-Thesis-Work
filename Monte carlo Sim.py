import numpy as np
import matplotlib.pyplot as plt
import pandas

from ripser import ripser
from tqdm import tqdm

# import multiprocessing
# from persim import plot_diagrams




def fibonacci_sphere(samples):
    '''
    Input: integer n for sample size
    Process: generate a point cloud of the sample size that is uniformly distributed on the surface of a
        sphere
    Output: The point cloud array, so it's an n x 3 array
    '''
    indices = np.arange(samples, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / samples)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi) # Each row vectors

    return np.column_stack((x, y, z))

# Generate point cloud
# num_points = 1000

 # 1000 x 3 matrix


def apply_radial_noise(num_points):
    '''
    Inputs: integer n, for sample size
    Process: Call the fibonacci function for an n x 3 array
    Output: The noisy spherical  n x 3 array
    '''
    points = fibonacci_sphere(num_points)
    # Apply radial noise
    mean = 1
    std_dev = 0.05
    noise = np.random.normal(mean, std_dev, num_points)
    points *= noise[:, np.newaxis]  # Scale radius by noise
    return points


# diagrams = ripser(apply_radial_noise(num_points), maxdim=2)['dgms']

# Plot the persistence diagrams
# plot_diagrams(diagrams)
# plt.show()

# Calculate the lifetime of the fundamental class
# [[x,y]] = diagrams[2]
# lifetime = y - x
# print(lifetime)

# print(diagrams[2])
# print(len(diagrams[2]))

# Run a monte carlo simulation

'''
Function that takes in a pt_cloud and outputs the  lifetime of the fundamental class.
       Have to ensure that all iterations only contain a single point in H2, otherwise find the max lifetime. Check
       if the length of the array is greater than 1 (row) if it is, run a loop ranging over the length of the array
       calculating the lifetime of each row in the array and place these values in an array. Add a column to this array 
       of integers from 0 to the length of the array so now you have an nx2 array. Order the array from greatest to 
       lowest lifetimes, then select the the row of the diagrams array that is the integer value in the 0'th row after
       ordering the lifetime's array.
'''

def get_fund_lifetime(pt_cloud):
    '''
    Inputs: Pt_cloud array
    Process: Run a persistance homology algortihm on the pt cloud and determines the lifetime of the
        fundamental homology class. (detects the presence of a void in the pt_cloud's shape, and lifetime
        represents the
    '''
    diagrams = ripser(pt_cloud, maxdim=2)['dgms']
    lifetime = max(abs(np.diff(diagrams[-1], axis=1)))[0]
    return lifetime

'''
 Initialize the array of 2 dimensions, the number of datapoints n, and the lifetime of the fundamental class for each
 iteration
'''
n_lifetime_df = np.zeros((1000, 2))

for i in tqdm(range(10)):
    num_points = 100 + 100*i
    for j in range(100):
        pt_cloud = apply_radial_noise(num_points)
        lifetime = get_fund_lifetime(pt_cloud)
        n_lifetime_df[j+100*i, 0] = num_points
        n_lifetime_df[j+100*i, 1] = lifetime


# for j in tqdm(range(100)):
#     pt_cloud = apply_radial_noise(1000)
#     lifetime = get_fund_lifetime(pt_cloud)
#     n_lifetime_df[j, 0] = 1000
#     n_lifetime_df[j, 1] = lifetime


n_lifetime_df = pandas.DataFrame(data = n_lifetime_df, columns = ['Sample size', 'Lifetime of Fundamental Class'])

print(n_lifetime_df)


n_lifetime_df.boxplot(by='Sample size', column =['Lifetime of Fundamental Class'],
                                grid=False)

# plt.axhline(y = np.sqrt(2), color='r', linestyle = '-')

plt.show()
