import numpy as np

# i = np.arange(0,2467+1)
# terms = (0 + 1/2468*i)**5 * (1/2468)
#
# value = sum(terms)
# print(value)


def process(right):
    '''
    Input: x, a scalar, the upper limit of integration for integral of cos(x) from 0 to x
    '''
    n = 8000
    x = np.linspace(0,right,n+1)
    func_values = np.cos(x)
    dx = x[1] - x[0]
    areas = dx * func_values[:-1]

    integral = sum(areas)
    return integral

a = np.linspace(0,4*np.pi, 200)

results = np.zeros(len(a))

for i in range(len(a)):
    results[i] = process(a[i])

print(results)

from matplotlib import pyplot as plt


fig,ax = plt.subplots()

plt.scatter(a,results)

plt.show()
