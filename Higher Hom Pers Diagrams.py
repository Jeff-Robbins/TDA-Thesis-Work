import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
import multiprocessing
from tqdm import tqdm
import pandas





def fibonacci_sphere(samples):
    indices = np.arange(samples, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / samples)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi) # Each row vectors

    return np.column_stack((x, y, z)) # 3 x 1000 matrix

# Generate point cloud
# num_points = 1000

 # 1000 x 3 matrix




def get_fund_lifetime(pt_cloud):
    diagrams = ripser(pt_cloud, maxdim=2)['dgms']
    lifetime = max(abs(np.diff(diagrams[-1], axis=1)))[0]
    return lifetime

def apply_radial_noise(num_points):
    points = fibonacci_sphere(num_points)
    # Apply radial noise
    mean = 1
    std_dev = 0.05
    noise = np.random.normal(mean, std_dev, num_points)
    points *= noise[:, np.newaxis]  # Scale radius by noise
    return points

def process_data(args):
    i, num_points = args
    pt_cloud = apply_radial_noise(num_points)
    lifetime = get_fund_lifetime(pt_cloud)
    return i, num_points, lifetime

# def populate_df(n):
#     n_lifetime_df = np.zeros((n, 2))
#     pool = multiprocessing.Pool()
#     args_list = [(j+100*i, 100 + 100*i) for i in range(20) for j in range(100)]
#     results = list(tqdm(pool.imap(process_data, args_list), total=n))
#     for i, num_points, lifetime in results:
#         n_lifetime_df[i, 0] = num_points
#         n_lifetime_df[i, 1] = lifetime
#     return n_lifetime_df


# if __name__ == '__main__':
#     n = 2000
#     pool = multiprocessing.Pool()
#     results = list(tqdm(pool.imap(process_data, range(100, 2100, 100)), total=n))

def populate_df(n):
    n_lifetime_df = np.zeros((n, 2))
    pool = multiprocessing.Pool()  # Create a multiprocessing pool
    results = []
    for i in tqdm(range(5)):
        num_points = 100 + 100*i
        for j in range(5):
            pt_cloud = apply_radial_noise(num_points)
            # Use the multiprocessing pool to parallelize the computation of lifetime
            result = pool.apply_async(get_fund_lifetime, (pt_cloud,))
            results.append(result)
            n_lifetime_df[j+5*i, 0] = num_points
    # Retrieve the results from the multiprocessing pool
    for i, result in enumerate(results):
        lifetime = result.get()
        n_lifetime_df[i, 1] = lifetime
    pool.close()
    pool.join()
    return n_lifetime_df

n_lifetime_df = populate_df(25)



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



'''
 Initialize the array of 2 dimensions, the number of datapoints n, and the lifetime of the fundamental class for each
 iteration
'''


# def populate_df(n):
#     n_lifetime_df = np.zeros((n, 2))
#     for i in tqdm(range(20)):
#         num_points = 100 + 100*i
#         for j in range(100):
#             pt_cloud = apply_radial_noise(num_points)
#             lifetime = get_fund_lifetime(pt_cloud)
#             n_lifetime_df[j+100*i, 0] = num_points
#             n_lifetime_df[j+100*i, 1] = lifetime
#     return n_lifetime_df
#
# n_lifetime_df = populate_df(2000)

# print(n_lifetime_df)



x = n_lifetime_df[:, 0]
y = n_lifetime_df[:, 1]

# Create scatter plot
plt.scatter(x, y, label='Data')

plt.xlabel('Number of data points in point cloud')
plt.ylabel('Lifetime of fundamental homology class')


n_lifetime_df = pandas.DataFrame(data = n_lifetime_df, columns = ['n','lifetime'])

print(n_lifetime_df)


n_lifetime_df.boxplot(by = 'n', column =['lifetime'],
                                grid = False)

plt.axhline(y = np.sqrt(3), color = 'r', linestyle = '-')

plt.show()

# Fit a line to the data
# fit = np.polyfit(x, y, 1)
# line = np.poly1d(fit)
# x_fit = np.linspace(0, 1, 100)
# y_fit = line(x_fit)
#
# # Plot the best fit line
# plt.plot(x_fit, y_fit, color='red', label='Best Fit Line')
#
# # Add labels and legend
# # plt.xlabel('x')
# # plt.ylabel('y')
# plt.legend()
#
# # Show the plot
# plt.show()




