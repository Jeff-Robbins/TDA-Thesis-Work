import numpy as np
from ripser import ripser


def fibonacci_sphere(samples):
    indices = np.arange(samples, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / samples)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return np.column_stack((x, y, z))


def apply_radial_noise(num_points):
    points = fibonacci_sphere(num_points)
    mean = 1
    std_dev = 0.05
    noise = np.random.normal(mean, std_dev, num_points)
    points *= noise[:, np.newaxis]
    return points

def get_fund_lifetime(pt_cloud):
    diagrams = ripser(pt_cloud, maxdim=2)['dgms']
    lifetime = max(abs(np.diff(diagrams[-1], axis=1)))[0]
    return lifetime

def process_data(num_points):
    pt_cloud = apply_radial_noise(num_points)
    lifetime = get_fund_lifetime(pt_cloud)
    return (num_points, lifetime)
