import fractal_clusters.fractal as frac
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import time
import sys
import os
import gc

################# Parameters
nr_walkers = int(10e3) #number of walkers
n = '4' #change string number if want simulation with same number of walkers without overwriting
#################

dirS = 'dla_data'
if not os.path.exists(dirS):
    os.makedirs(dirS)
try:
    PLOT = str(sys.argv[1])
except:
    PLOT = 'F'
if PLOT != 'P':
    time1 = time.time()
    pos = frac.DLA_cluster(nr_walkers)
    time2 = time.time()

    print(f'Time elapsed: {time2-time1:.2f} seconds')
    np.save("%s/dla_pos%s N=%.1f.npy" % (dirS, n, nr_walkers), pos)

else:
################# Parameters
    rainbow = True
    colorful = False
    blue = False
    cluster = np.load("%s/dla_pos%s N=%.1f.npy" % (dirS, n, nr_walkers))
#################
    L = np.max(abs(cluster))
    size = cluster[:,0].shape[0]

    plt.title('%.f Walkers' % (nr_walkers))
    plt.xlim([-1.2*L,1.2*L])
    plt.ylim([-1.2*L,1.2*L])
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

    if colorful:
        plt.scatter(cluster[:,0], cluster[:,1], s=4, c=np.random.rand(len(cluster[:,0]),3), marker="*")
        plt.savefig('DLA_cluster_%.f_walkers_colorful.png' % (nr_walkers))
    if rainbow:
        colors = cm.rainbow(np.linspace(0,1, size))
        plt.scatter(cluster[:,0], cluster[:,1], s=4, c=colors, marker="*")
        plt.savefig('DLA_cluster%s_%.f_walkers_rainbow.png' % (n,nr_walkers))
    if blue:
        plt.scatter(cluster[:,0], cluster[:,1], s=4, marker="*")
        plt.savefig('DLA_cluster_%.f_walkers.png' % (nr_walkers))

    plt.show()
