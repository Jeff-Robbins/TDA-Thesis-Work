import matplotlib.pyplot as plt
import numpy as np
from ripser import ripser
from persim import plot_diagrams

# Generate a random 3-dimensional point cloud
point_cloud = np.random.rand(100, 3)

# Compute the persistence diagrams
diagrams = ripser(point_cloud, maxdim=2)['dgms']

# Plot the persistence diagrams
plot_diagrams(diagrams)
plt.show()