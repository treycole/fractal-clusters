import fractal_clusters.fractal as frac
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# this file determines the density density correlation for 4 N=10000 DLA clusters and fits a power law
N = 10000

#load the four clusters
pos1 = np.load("dla_data/dla_pos N=10000.0.npy")
pos2 = np.load("dla_data/dla_pos2 N=10000.0.npy")
pos3 = np.load("dla_data/dla_pos3 N=10000.0.npy")
pos4 = np.load("dla_data/dla_pos3 N=10000.0.npy")

#calculate density distribution for all clusters individually
r1, dens1, = frac.density(pos1,N)
r2, dens2, = frac.density(pos2,N)
r3, dens3, = frac.density(pos3,N)
r4, dens4, = frac.density(pos4,N)

#plot of the densities
r_max = int(min(np.amax(r1),np.amax(r2),np.amax(r3),np.amax(r4)))
r_avg = np.linspace(0,r_max,r_max+1)
dens_avg = np.zeros(r_max+1)
dens_std = np.zeros(r_max+1)

for i in range(r_max+1):
    dens_avg[i] = (dens1[i]+dens2[i]+dens3[i]+dens4[i])/4
    dens_std[i] = np.std(np.array([dens1[i], dens2[i], dens3[i], dens4[i]]))

min_r_fit = 5
max_r_fit = int(r_max*0.6666)
alpha, beta = optimize.curve_fit(frac.func, xdata = r_avg[min_r_fit :max_r_fit], ydata = dens_avg[min_r_fit:max_r_fit]) #alpha contains fitted par in func, beta is cov matrix
std = np.sqrt(np.diag(beta)) #derive std of paramterers

y = frac.func(r_avg,alpha[0],alpha[1]) #fit
print(f'The fitted parameters of y = a*r**b are: ')
print(f'a={alpha[0]:3f}+-{std[0]:3f}')
print(f'b={alpha[1]:3f}+-{std[1]:3f}')

plt.figure(figsize=(8, 6))
plt.errorbar(r_avg, dens_avg, yerr=dens_std, fmt="o", markersize=2, color='red', ecolor = 'black')
plt.plot(r_avg,y,label='Fitted Curve A=%.3f'%(abs(alpha[1])),linestyle='-.')
plt.grid()
plt.legend()
plt.xlabel('Radius',size=12)
plt.ylabel('Density',size=12)
plt.xscale("log")
plt.yscale("log")
plt.xlim([0.99999,r_max])
plt.tight_layout()
plt.savefig('figs/DLA-Density.png')
plt.show()
