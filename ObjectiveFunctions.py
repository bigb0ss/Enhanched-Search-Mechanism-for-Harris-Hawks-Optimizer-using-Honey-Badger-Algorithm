import numpy as np

# Unimodal function

# Chung Reynolds


def f1(x):
    return np.sum(x**2)**2


# De Jong's (sphere)
def f2(x):
    return np.sum(x**2)


# Powell Sum
def f4(x):
    return sum([np.abs(x[i])**i+1 for i in range(len(x))])


# Schwefel 2.23
def f7(x):
    return np.sum(x**10)


# Multimodal Functions

# Brown
def f10(x):
    return sum([(x[i]**2)**(x[i+1]**2 + 1) + (x[i+1]**2)**(x[i]**2 + 1) for i in range(len(x)-1)])


# Griewank
def f13(x):
    return 1 + np.sum(x**2 / 4000) - np.product([np.cos(x[i])/np.sqrt(i+1) for i in range(len(x))])


# Zakharov
def f16(x):
    return np.sum(x**2) + np.sum([0.5*x[i]*(i+1) for i in range(len(x))]) ** 2 + np.sum([0.5*x[i]*(i+1) for i in range(len(x))]) ** 4
