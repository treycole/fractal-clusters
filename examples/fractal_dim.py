import fractal_clusters.fractal as frac
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

import time
import sys
import os

################# Parameters
nr_walkers = int(10e3) #number of walkers
#################
dirS = 'dla_data'

pos1 = np.load("dla_data/dla_pos N=10000.0.npy")
pos2 = np.load("dla_data/dla_pos2 N=10000.0.npy")
pos3 = np.load("dla_data/dla_pos3 N=10000.0.npy")
pos4 = np.load("dla_data/dla_pos3 N=10000.0.npy")

rad1 = np.sqrt(frac.euc_dist_sq(pos1[:,0], pos1[:,1]))
rad2 = np.sqrt(frac.euc_dist_sq(pos2[:,0], pos2[:,1]))
rad3 = np.sqrt(frac.euc_dist_sq(pos3[:,0], pos3[:,1]))
rad4 = np.sqrt(frac.euc_dist_sq(pos4[:,0], pos4[:,1]))

#plot of the densities
r_max = int(min(np.amax(rad1),np.amax(rad2),np.amax(rad3),np.amax(rad4)))
r = np.linspace(0,r_max,r_max+1)
c = np.zeros((4, r.shape[0]))

for j in range(r.shape[0]):
    c[0, j] = frac.c(pos1, r[j])
    c[1, j] = frac.c(pos2, r[j])
    c[2, j] = frac.c(pos3, r[j])
    c[3, j] = frac.c(pos4, r[j])

c_avg = np.sum(c, axis=0)/4
c_std = np.zeros(r.shape[0])
for i in range(r.shape[0]):
    c_avg[i] = (c[0,i]+c[1,i]+c[2,i]+c[3,i])/4
    c_std[i] = np.std(np.array([c[0,i], c[1,i], c[2,i], c[3,i]]))

min_r_fit = 5
max_r_fit = int(r_max*0.6666)
[k,dim], beta = optimize.curve_fit(frac.func, xdata = r[min_r_fit :max_r_fit], ydata = c_avg[min_r_fit :max_r_fit]) #alpha contains fitted par in func, beta is cov matrix
std = np.sqrt(np.diag(beta)) #derive std of par


print(f'The fitted parameters of y = k*r**d are: ')
print(f'k={k:3f}+-{std[0]:3f}')
print(f'd={dim:3f}+-{std[1]:3f}')

plt.figure(figsize=(8, 6))
plt.errorbar(r, c_avg, yerr=c_std, label='Number of particles in circle of radius r', fmt="o", markersize=2, color='red', ecolor = 'black')
plt.plot(r, frac.func(r, k, dim), label='Fitted Curve d=%.3f'%(dim), c='b', linestyle='-.')
plt.grid()
plt.legend()
plt.xlabel('Radius',size=12)
plt.ylabel('N(r)',size=12)
plt.tight_layout()
plt.savefig('figs/frac_dim.png')
plt.show()
