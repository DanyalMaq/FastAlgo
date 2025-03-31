import numpy as np
from timming import timeit

@timeit
def cost_function(points, labels, centers):
    assignment = np.array([centers[labels[i]] for i in labels])
    cost = np.power(points-assignment, 2).sum(axis=1).sum()
    return cost

@timeit
def cost_function_2(points, labels, centers):
    assignment = np.array([map(lambda x: centers[x], labels)])
    cost = np.power(points-assignment, 2).sum(axis=1).sum()
    return cost