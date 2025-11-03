import fractal as frac
import numpy as np
import matplotlib.pyplot as plt

import time
import sys
import os
import gc

# File used to calculate potential profile of a static cluster grown outside of an environment with any voltage differences at the center of a lead with the cluster grounded

num_pts = [500, 1000, 2000, 3000, 5000, 10000, 20000, 50000, 100000]
n = 5
# True False
DLA = True
Eden = False

if DLA:
    dirS = 'dla_data'
    cluster = np.load("%s/dla_pos N=%.1f.npy"%(dirS, num_pts[n]))
elif Eden:
    dirS = 'eden_data'
    cluster = np.load("%s/eden_cluster N=%.1f.npy" % (dirS, num_pts[n]))

max_r = np.amax(np.sqrt(frac.euc_dist_sq(cluster[:,0], cluster[:,1])))
r_lead = max_r+2
BC_lead = 1
BC_cluster = 0

dirS = 'dielectric_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    phi = frac.Laplace(r_lead, cluster, BC_lead=BC_lead, BC_cluster=BC_cluster, static=True)
    np.save("%s/phi N=%.1f r_lead=%.1f BC_lead=%.1f BC_cluster=%.1f.npy"%(dirS, num_pts[n], r_lead, BC_lead, BC_cluster), phi)
else:
    phi = np.load("%s/phi N=%.1f r_lead=%.1f BC_lead=%.1f BC_cluster=%.1f.npy"%(dirS, num_pts[n], r_lead, BC_lead, BC_cluster))
    plt.imshow(phi, origin='lower')
    delta = int((phi.shape[0]-1)/2) #shift origin of cluster
    plt.scatter(cluster[:,0]+delta, cluster[:,1]+delta, c='k')
    plt.colorbar()
    plt.savefig('potential N=%.1f r_lead=%.1f BC_lead=%.1f BC_cluster=%.1f.png'%(num_pts[n], r_lead, BC_lead, BC_cluster))
    plt.show()
