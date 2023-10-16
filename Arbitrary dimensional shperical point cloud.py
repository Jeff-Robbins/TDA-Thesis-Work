import numpy as np
import pandas as pd


def generate_point_cloud(num_points, dimensions):
    def fibonacci_sphere(samples):
        indices = np.arange(samples, dtype=float) + 0.5

        phi = np.arccos(1 - 2 * indices / samples)
        theta = np.pi * (1 + 5 ** 0.5) * indices

        coordinates = []
        for i in range(dimensions - 1):
            coord = np.cos(theta) * np.sin(phi)
            coordinates.append(coord)

        coordinates.append(np.cos(phi))

        return np.column_stack(coordinates)

    # Generate point cloud
    points = fibonacci_sphere(num_points)

    # Apply radial noise
    mean = 1
    std_dev = 0.05
    noise = np.random.normal(mean, std_dev, num_points)
    points *= noise[:, np.newaxis]  # Scale radius by noise

    # Convert points to DataFrame
    columns = [f"Dimension {i + 1}" for i in range(dimensions)]
    df = pd.DataFrame(points, columns=columns)

    return df


# Generate a point cloud with 1000 points in 4 dimensions
num_points = 1000
dimensions = 4
point_cloud = generate_point_cloud(num_points, dimensions)

# Print the DataFrame of the point cloud
print(point_cloud)
