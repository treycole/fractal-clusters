import animations as ani
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class LoopingPillowWriter(PillowWriter):
    def finish(self):
        self._frames[0].save(
            self._outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps), loop=0)
##############
"""
dirS = 'dla_data'
n =''
nr = 1000
cluster = np.load("%s/dla_pos%s N=%.1f.npy" % (dirS, n, nr))[:-2]
animation = ani.animate_cluster(cluster)
plt.show()
animation.save('gifs/DLA_cluster_%s_animation_%.0f.gif'%(n, nr), writer=LoopingPillowWriter(fps=20))
"""
############
"""
dirS = 'eden_data'
nr = 950
cluster = np.load("%s/eden_cluster N=%.1f.npy" % (dirS, nr))
perimeter = np.load("%s/eden_perim N=%.1f.npy" % (dirS, nr))

animation = ani.animate_cluster(cluster)
plt.show()
animation.save('gifs/Eden_cluster_animation_%.0f.gif'%(nr), writer=LoopingPillowWriter(fps=20))
"""
############

dirS = 'dielectric_data'
nr = 1000
eta = 1
BC_lead = 1
BC_cluster = 0
r_lead = 200

cluster= np.load('%s/Eden_cluster n_cells=%.0f eta=%.0f BC_lead=%.1f BC_cluster=%.1f r_lead=%.1f.npy' % (dirS, nr, eta, BC_lead, BC_cluster, r_lead))[:-2]
perimeter = np.load('%s/Eden_perimeter n_cells=%.0f eta=%.0f BC_lead=%.1f BC_cluster=%.1f r_lead=%.1f.npy' % (dirS, nr, eta, BC_lead, BC_cluster, r_lead))[:-2]
phi = np.load('%s/Eden_phi n_cells=%.0f eta=%.0f BC_lead=%.1f BC_cluster=%.1f r_lead=%.1f.npy' % (dirS, nr, eta, BC_lead, BC_cluster, r_lead))

animation = ani.animate_cluster(cluster, phi=phi)
plt.show()
animation.save('gifs/dielectric_cluster_animation_%.0f_%.0f_%.1f.gif'%(nr, eta, r_lead), writer=LoopingPillowWriter(fps=20))
