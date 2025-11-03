import fractal_clusters.fractal as frac
import numpy as np
import matplotlib.pyplot as plt

steps = np.linspace(0, 20)
x_coor, y_coor = np.zeros(steps.shape[0]), np.zeros(steps.shape[0])
x_coor[0], y_coor[0] = 0,0
for i in range(0, steps.shape[0]-1):
    x_i, y_i = x_coor[i], y_coor[i]
    if i == 0:
        x_prev, y_prev = x_i, y_i
    else:
        x_prev, y_prev = x_coor[i-1], y_coor[i-1]

    x_coor[i+1], y_coor[i+1] = frac.random_walker(x_i, y_i, x_prev, y_prev, lattice_size=20)

pos = [x_coor, y_coor]
print(pos[0])
print(np.amax(abs(pos[0][:])))
frac.animate_walker(pos)
