# Basic imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from IPython.display import Video


# pip install teaspoon
# pip install ripser
# pip install persim



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



r =1
R =2
P = makePtCloud.Annulus(N=200,r=r, R=R, seed=None) # Teaspoon data generation
plt.scatter(P[:,0],P[:,1])
print(P)


def drawTDAtutorial(P, diagrams, R=2):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

    # Draw point cloud
    plt.sca(axes[0])
    plt.title('Point Cloud')
    plt.scatter(P[:, 0], P[:, 1])

    # Draw diagrams
    plt.sca(axes[1])
    plt.title('0-dim Diagram')
    Draw.drawDgm(diagrams[0])

    plt.sca(axes[2])
    plt.title('1-dim Diagram')
    Draw.drawDgm(diagrams[1])
    plt.axis([0, R, 0, R])

diagrams = ripser.ripser(P)['dgms']
drawTDAtutorial(P,diagrams)


# Double annulus

def DoubleAnnulus(r1 = 1, R1 = 2, r2 = 0.8, R2 = 1.3, xshift = 3):
    P = makePtCloud.Annulus(r = r1, R = R1)
    Q = makePtCloud.Annulus(r = r2, R = R2)
    Q[:,0] = Q[:,0] + xshift
    P = np.concatenate((P,Q) )
    return(P)

P = DoubleAnnulus(r1 = 1, R1 = 2, r2 = 1, R2 = 1.3, xshift = 3)
diagrams = ripser.ripser(P)['dgms']


# Draw Ripser persidtence diagrams
drawTDAtutorial(P,diagrams, R=2.5)
