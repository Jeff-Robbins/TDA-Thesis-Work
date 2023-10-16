# import numpy as np
# import matplotlib.pyplot as plt
import persim.landscapes
import tadasets
import persim

from ripser import ripser

loop = tadasets.dsphere(d=1)
dgms = ripser(loop)
persim.plot_diagrams(dgms)