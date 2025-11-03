import fractal_clusters.fractal as frac
import numpy as np
import matplotlib.pyplot as plt

import time
import sys
import os

nr_cells = 1000
eta = 1
BC_lead = 1
BC_cluster = 0
r_lead = 200#2*nr_cells

dirS = 'dielectric_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':

    cluster, perimeter, phi = frac.Eden_dielec_cluster(nr_cells, eta, r_lead, BC_lead=BC_lead, BC_cluster=BC_cluster)

    np.save('%s/Eden_cluster n_cells=%.0f eta=%.0f BC_lead=%.1f BC_cluster=%.1f r_lead=%.1f.npy' % (dirS, nr_cells, eta, BC_lead, BC_cluster, r_lead), cluster)
    np.save('%s/Eden_perimeter n_cells=%.0f eta=%.0f BC_lead=%.1f BC_cluster=%.1f r_lead=%.1f.npy' % (dirS, nr_cells, eta, BC_lead, BC_cluster, r_lead), perimeter)
    np.save('%s/Eden_phi n_cells=%.0f eta=%.0f BC_lead=%.1f BC_cluster=%.1f r_lead=%.1f.npy' % (dirS, nr_cells, eta, BC_lead, BC_cluster, r_lead), phi)
else:
    cluster = np.load('%s/Eden_cluster n_cells=%.0f eta=%.0f BC_lead=%.1f BC_cluster=%.1f r_lead=%.1f.npy' % (dirS, nr_cells, eta, BC_lead, BC_cluster, r_lead))
    perimeter = np.load('%s/Eden_perimeter n_cells=%.0f eta=%.0f BC_lead=%.1f BC_cluster=%.1f r_lead=%.1f.npy' % (dirS, nr_cells, eta, BC_lead, BC_cluster, r_lead))
    phi = np.load('%s/Eden_phi n_cells=%.0f eta=%.0f BC_lead=%.1f BC_cluster=%.1f r_lead=%.1f.npy' % (dirS, nr_cells, eta, BC_lead, BC_cluster, r_lead))

    plt.imshow(phi)
    plt.colorbar()
    plt.scatter(cluster[:,0], cluster[:,1], c='b', marker='o', s=5)
    plt.scatter(perimeter[:,0], perimeter[:,1], c='r', marker='o', s=2)
    plt.savefig('figs/dielectric_breakdown_sim.png')
    plt.show()
