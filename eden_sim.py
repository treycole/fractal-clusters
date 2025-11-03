import fractal as frac
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import time
import sys
import os
import gc

nr_cells = int(950)

dirS = 'eden_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    time1 = time.time()
    cluster, perimeter = frac.Eden_cluster(nr_cells)
    time2 = time.time()

    print(f'Time elapsed: {time2-time1:.2f} seconds')
    np.save("%s/eden_cluster N=%.1f.npy" % (dirS, nr_cells), cluster)
    np.save("%s/eden_perim N=%.1f.npy" % (dirS, nr_cells), perimeter)
else:
    rainbow = True
    colorful = False
    red_and_blue = False

    cluster = np.load("%s/eden_cluster N=%.1f.npy" % (dirS, nr_cells))
    perimeter = np.load("%s/eden_perim N=%.1f.npy" % (dirS, nr_cells))

    Lx = 1.2*max(perimeter[:,0])
    Ly = 1.2*max(perimeter[:,1])

    plt.title('%.f Cells' % (nr_cells))
    plt.xlim(-Lx, Lx)
    plt.ylim(-Ly, Ly)
    plt.tick_params(
        axis='both', #x or y axis
        which='both', #major or minor ticks
        bottom=False, #ticks are on or off on bottom
        top=False, #ticks are on or off on top
        left=False, #ticks are on or off on left
        right=False, #ticks are on or off on right
        labelbottom=False, #label on or off bottom
        labeltop=False, #label on or off top
        labelleft=False, #label on or off left
        labelright=False #label on or off right
    )

    if rainbow:
        colors = cm.rainbow(np.linspace(0,1, nr_cells))
        plt.scatter(cluster[:, 0], cluster[:, 1], c=colors, s=5, marker='.')
        plt.scatter(perimeter[:, 0], perimeter[:, 1], s=4, edgecolors='b', facecolors='None', marker='o')
        plt.savefig('Eden_cluster_%.f_cells_rainbow.png' % (nr_cells))
    if red_and_blue:
        plt.scatter(cluster[:, 0], cluster[:, 1], c='r', s=5, marker='.')
        plt.scatter(perimeter[:, 0], perimeter[:, 1], s=4, edgecolors='b', facecolors='None', marker='o')
        plt.savefig('Eden_cluster_%.f_cells.png' % (nr_cells))
    if colorful:
        plt.scatter(cluster[:, 0], cluster[:, 1], c = np.random.rand(len(cluster[:,0]), 3), marker='x')
        plt.scatter(perimeter[:, 0], perimeter[:, 1], c='b', s=8)
        plt.savefig('Eden_cluster_%.f_cells_colorful.png' % (nr_cells))

    plt.show()
