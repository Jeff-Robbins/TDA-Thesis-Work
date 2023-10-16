import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import uniform
from ripser import ripser
from persim import plot_diagrams
import pandas as pd
import tqdm


def generate_point_cloud(num_points, noise_mean, noise_std):
    # Generate uniform random points on the surface of a unit sphere
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(num_points)
    z = np.linspace(1 - 1 / num_points, 1 / num_points - 1, num_points)
    radius = np.sqrt(1 - z ** 2)

    # Add radial noise to the points
    noise = uniform.rvs(loc=noise_mean, scale=noise_std, size=num_points)
    radius += noise

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    point_cloud = np.column_stack((x, y, z))
    return point_cloud


# Set the number of points and noise parameters
num_points = 100
noise_mean = 1.0
noise_std = 0.1

# Generate the point cloud
point_cloud = generate_point_cloud(num_points, noise_mean, noise_std)

# Compute persistence diagram using ripser
diagrams = ripser(point_cloud)['dgms']

def get_fund_lifetime(pt_cloud):
    '''
    Inputs: Pt_cloud array
    Process: Run a persistance homology algortihm on the pt cloud and determines the lifetime of the
        fundamental homology class. (detects the presence of a void in the pt_cloud's shape, and lifetime
        represents the
    '''
    diagrams = ripser(pt_cloud)['dgms']
    lifetime = max(abs(np.diff(diagrams[-1], axis=1)))[0]
    return lifetime



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


n_lifetime_df = pd.DataFrame(data = n_lifetime_df, columns = ['Sample size', 'Lifetime of Fundamental Class'])

print(n_lifetime_df)


n_lifetime_df.boxplot(by='Sample size', column =['Lifetime of Fundamental Class'],
                                grid=False)

# plt.axhline(y = np.sqrt(2), color='r', linestyle = '-')

plt.show()






# # Plot the point cloud
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=5)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Point Cloud')
#
# # Plot persistence diagrams using persim
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plot_diagrams(diagrams, plot_only=[0])
# plt.title('Persistence Diagram (H0)')
#
# plt.subplot(1, 2, 2)
# plot_diagrams(diagrams, plot_only=[1])
# plt.title('Persistence Diagram (H1)')
#
# plt.show()
