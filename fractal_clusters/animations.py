import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.widgets import Slider
from matplotlib import cm

def euc_dist_sq(x, y):
    """
    Calculates the squared Euclidean distance

    Parameters
    ----------
    x : np.ndarray
        x-coordinates
    y : np.ndarray
        y-coordinates

    Returns
    ----------
    dist_sq : np.ndarray
        squared Euclidean distance

    """
    dist_sq = x**2 + y**2
    return dist_sq

def animate_cluster(cluster, slider=False, text=False, border=False, interval=1, perimeter=None, phi=None):
    """
    Animates the cluster growth

    Parameters
    ----------
    positions: np.ndarray
        positions of the random walker at each step
    slider : boolean
        whether or not the animation changes frames automatically (False) or manually (True)
    interval : integer
        speed of frame updates when slider = False

    Returns
    ----------

    """
    fig, ax = plt.subplots()
    size = cluster[:,0].shape[0]
    colors = cm.rainbow(np.linspace(0,1, size))
    r = np.sqrt(euc_dist_sq(cluster[:,0], cluster[:,1]))
    r_max = r.max()
    ax.scatter(cluster[0,0], cluster[0,1], c='k', s=30, marker="x", zorder=2)

    ax.set_xlim(left=-r_max, right=r_max)
    ax.set_ylim(bottom=-r_max, top=r_max)
    if not border:
        ax.axis("off")
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
    txt = fig.suptitle('')

    if not phi is None:
        ax.set_xlim(left=0, right=phi.shape[0])
        ax.set_ylim(bottom=0, top=phi.shape[0])
        plt.imshow(phi)
        plt.colorbar()

    if not perimeter is None:
        image_clust = ax.scatter(cluster[0,0], cluster[0,1], s=10, marker="o")
        image_perim = ax.scatter(perimeter[:,0], perimeter[:,1], c='k', marker='s', s=2)

        def update(i):
            i = int(i)
            image_clust.set_offsets(cluster[:i, :])
            image_clust.set_color(colors[:i])
            image_perim.set_offsets(perimeter[:i, :])
            if text:
                txt.set_text(r'Cluster point %.1f' % i)
                return image_clust, image_perim, txt
            return image_clust, image_perim

    else:
        image = ax.scatter(cluster[0,0], cluster[0,1], s=10, marker="o")
        def update(i):
            i = int(i)
            image.set_offsets(cluster[:i, :])
            image.set_color(colors[:i])
            if text:
                txt.set_text(r'Cluster point %.1f' % i)
                return image, txt
            return image

    if slider:
        axstep = plt.axes([0.25, .03, 0.50, 0.02])
        slider_bar = Slider(axstep, label='step', valmin=0, valmax=size-1, valfmt='%0.0f', valstep=1)
        slider_bar.on_changed(update)
        return

    elif not slider:
        anim = animation.FuncAnimation(fig, update, frames=size-1, interval=interval)
        return anim

def animate_walker(positions, slider=False, interval=500):
    """
    Animates the random walker

    Parameters
    ----------
    positions: np.ndarray
        positions of the random walker at each step
    slider : boolean
        whether or not the animation changes frames automatically (False) or manually (True)
    interval : integer
        speed of frame updates when slider = False

    Returns
    ----------

    """
    fig, ax = plt.subplots()
    image = ax.scatter(positions[0][0], positions[1][0])
    ax.set_xlim(left=-np.amax(abs(positions[0][:])), right=np.amax(abs(positions[0][:])))
    ax.set_ylim(bottom=-np.amax(abs(positions[1][:])), top=np.amax(abs(positions[1][:])))
    txt = fig.suptitle('')
    def update(i):
        i = int(i)
        txt.set_text(r'Walker step %.1f' % i)
        image.set_offsets([positions[0][i], positions[1][i]])
        return image, txt
    if slider:
        axstep = plt.axes([0.25, .03, 0.50, 0.02])
        slider_bar = Slider(axstep, label='step', valmin=0, valmax=positions[0].shape[0]-1, valfmt='%0.0f', valstep=1)
        slider_bar.on_changed(update)
    elif not slider:
        anim = animation.FuncAnimation(fig, update, frames=positions[0].shape[0]-1, interval=interval)
    plt.show()
    return
