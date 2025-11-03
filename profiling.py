import sys
from line_profiler import LineProfiler
import fractal as frac
import numpy as np


lprofiler = LineProfiler()

##################
"""
x_i = 5
y_i = 5
x_prev = 4
y_prev = 5
lattice_size = 20
lp_wrapper = lprofiler(frac.random_walker)
lp_wrapper(x_i, y_i, x_prev, y_prev, lattice_size)
"""
##################

N = 500
lp_wrapper = lprofiler(frac.DLA_cluster)
lp_wrapper(N)

##################
"""
n_cells = 100
r_lead = n_cells/2
lp_wrapper = lprofiler(frac.Eden_dielec_cluster)
lp_wrapper(n_cells, 1, r_lead)
"""
##################
"""
n_cells = 500
r_lead = n_cells/2
dirS = 'dla_data'
cluster = np.load("%s/dla_pos N=%.1f.npy"%(dirS, n_cells))
lp_wrapper = lprofiler(frac.Laplace)
lp_wrapper(r_lead, cluster)
"""
################

lprofiler.print_stats()
