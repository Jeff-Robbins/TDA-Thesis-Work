import numpy as np
import matplotlib.pyplot as plt
# Basic imports
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

def generate_point_cloud(num_points, mean_radius, std_radius):
    points = []
    for _ in range(num_points):
        theta = np.random.uniform(0,2*np.pi)
        radius = np.random.normal(mean_radius, std_radius)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append((x, y))
    return points

def generate_dataset(num_clouds, num_points_per_cloud, mean_radius, std_radius):
    dataset = []
    for _ in range(num_clouds):
        point_cloud = generate_point_cloud(num_points_per_cloud, mean_radius, std_radius)
        dataset.append(point_cloud)
    return dataset

# Generate a dataset of 10 point clouds, each with 1000 points on the unit circle with radial noise
dataset = generate_dataset(1, 1000, 1, 0.1)

# Display the first point cloud in the dataset
point_cloud = dataset[0]
point_cloud = np.array(point_cloud)


# Display the point cloud
fig, ax = plt.subplots()
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], c='b', marker='.')
ax.set_aspect('equal')
ax.set_xlim(-1.2,1.2)
ax.set_ylim(-1.2,1.2)

# Enable interactive rotation
def on_mouse_move(event):
    if event.inaxes == ax:
        ax.view_init(elev=30, azim=event.xdata)

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

print(dataset)


# diagrams = ripser.ripser(dataset)['dgms']

# Draw Ripser persistance diagram
# drawTDAtutorial(dataset,diagrams)
plt.show()