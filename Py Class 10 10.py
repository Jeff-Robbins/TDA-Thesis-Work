import numpy as np

def F(x):
    output = x - ((x**3) / 6) + ((x**5) / 120)
    return output

output = F(np.pi/2)

sin_x = np.sin(np.pi/2)

print("sin(pi/2) is ", sin_x)
print(output)



# Explicit sequence
def g(x,n):
    '''
    TBA...
    '''

    s = 0
    for k in range(0,n+1):
        s = s +
        k = np.arange(0, n + 1)
        num = (-1) ** k * x ** (2 * k + 1)

    denom =  np.math.factorial(2*k+1)
    a = num / denom
    vsl = sum(a)

    return val



