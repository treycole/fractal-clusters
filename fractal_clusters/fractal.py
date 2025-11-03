from re import I
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import matplotlib.cm as cm
from scipy.ndimage import convolve, generate_binary_structure
import random as rand
import sys
from scipy import optimize
from numpy import ones
import time

#np.random.seed(42)
def random_walker(x_i, y_i, x_prev, y_prev, lattice_size):
    """
    Updates the position of a walker undergoing random Brownian motion. Cannot go back from where it came.
    Uses periodic boundary conditions: if a walker leaves the box on the right, it enters on the left.

    Parameters
    -----------
    x_i : int
        initial x coordinate
    y_i : int
        initial y coordinate
    x_prev : int
        previous x coordinate
    y_prev : int
        previous y coordinate
    lattice_size : int
        determines size of box around origin

    Returns
    -----------
    x_f : int
        final x coordinate
    y_f : int
        final y coordinate
    """
    x_f = x_prev
    y_f = y_prev
    delta = np.random.choice([-1,1]) #forward or back
    while x_f == x_prev and y_f == y_prev:
        delta *= -1
        if np.random.choice([0,1]) == 0: #move along x
            x_f = (x_i + delta + lattice_size/2)%lattice_size - lattice_size/2
            y_f = (y_i + lattice_size/2)%lattice_size - lattice_size/2
        else: #move along y
            x_f = (x_i + lattice_size/2)%lattice_size - lattice_size/2
            y_f = (y_i + delta + lattice_size/2)%lattice_size - lattice_size/2

    return x_f, y_f

def DLA_cluster(nr_walkers, seed_pos=[0,0]):
    """
    Computes the positions of all the clustered random walkers using the diffusion limited aggregation technique.

    Parameters:
    ----------
    nr_walkers : int
        number of random walkers to be clustered
    seed_pos : list
        the coordinates of the seed position

    Returns:
    ----------
    cluster : np.ndarray
        the coordinates of all the clustered random walkers
    """
    cluster = np.tile(np.array(seed_pos),(nr_walkers,1)) #puts all the walkers at seed_pos initially
    count = 0
    for i in range(1, nr_walkers):
        print(nr_walkers-i)
        cling = 0
        r = np.sqrt(euc_dist_sq(cluster[:(i+1),0], cluster[:(i+1),1]))
        r_max = r.max()+1
        lattice_size = int(3*r_max) + int(3*r_max)%2  #box size, must be even

        #place walker at random edge
        if np.random.choice([0,1]) == 0:
            x_start = rand.randint(-lattice_size/2,lattice_size/2)
            y_start = -lattice_size/2
        else:
            x_start = -lattice_size/2
            y_start = rand.randint(-lattice_size/2,lattice_size/2)

        #keep moving the walker until it reaches the cluster
        x_prev, y_prev = x_start, y_start
        while cling == 0:
            count += 1
            x_new, y_new = random_walker(x_start, y_start, x_prev, y_prev, lattice_size)
            delta_pos_sum = abs(cluster[:(i+1), 0]-x_new) + abs(cluster[:(i+1),1]-y_new) #if cluster is neighbor, then only differs by one in either x or y but not both. So sum will = 1
            cond = delta_pos_sum[delta_pos_sum==1]
            if cond.shape[0] != 0:
                cluster[i,:] = [x_new, y_new]
                cling = 1
            else:
                x_prev, y_prev, x_start, y_start = x_start, y_start, x_new, y_new

    return cluster

def Eden_cluster(nr_cells, seed_pos=[0,0]):
    """
    Computes the positions of all the clustered random walkers using the Eden technique.

    Parameters:
    ----------
    nr_cells : int
        number of random walkers to be clustered
    eta : float
        determines how strongly chance of clustering depends on electric field
    seed_pos : list
        the coordinates of the seed position relative to the center of the ring
    Returns:
    ----------
    cluster : np.ndarray
        the coordinates of all the clustered random walkers
    perimeter : np.ndarray
        the perimeter of the cluster
    """

    center_of_latt = np.array(seed_pos)
    cluster = np.tile(center_of_latt,(nr_cells,1)) #start all cells at the center
    delta = np.array([[0,1],[1,0],[0,-1],[-1,0]]) #nearest neighbors
    perimeter = center_of_latt+delta
    for i in range(1, nr_cells):
        print(nr_cells-i)
        rand_idx = np.random.choice(perimeter.shape[0])
        cluster[i, :] = perimeter[rand_idx, :]
        perimeter = np.delete(perimeter, rand_idx, axis=0)
        candidates = cluster[i, :] + delta
        perimeter = np.append(perimeter, candidates[np.logical_and(~is_in_cluster(cluster, candidates), ~is_in_cluster(perimeter, candidates))], axis=0)

    return cluster, perimeter

def Eden_dielec_cluster(nr_cells, eta, r_lead, BC_lead=1, BC_cluster=0, seed_pos=[0,0]):
    """
    Computes the positions of all the clustered random walkers using the Eden technique.

    Parameters:
    ----------
    nr_cells : int
        number of random walkers to be clustered
    eta : float
        determines how strongly chance of clustering depends on electric field
    seed_pos : list
        the coordinates of the seed position relative to the center of the ring
    Returns:
    ----------
    cluster : np.ndarray
        the coordinates of all the clustered random walkers
    perimeter : np.ndarray
        the perimeter of the cluster
    """

    lattice_size = int(r_lead)*2+1+2 #lattice size should be the size of the radius of the ring, +1 for origin +2 for buffer
    shift = int((lattice_size-1)/2) #to shift origin to the center of the lattice
    center_of_latt = np.array(seed_pos)+shift
    cluster = np.tile(center_of_latt,(nr_cells,1)) #start all cells at the center
    delta = np.array([[0,1],[1,0],[0,-1],[-1,0]]) #nearest neighbors
    perimeter = center_of_latt+delta

    for i in range(1, nr_cells):
        pcnt_done = (1-((nr_cells-i)/nr_cells))*100
        if pcnt_done%10<1:
            print('%.0f pcnt' % (pcnt_done))

        phi = Laplace(r_lead, cluster, BC_lead=BC_lead, BC_cluster=BC_cluster)
        phi_perimeter = phi[perimeter[:,0], perimeter[:,1]] #phi indices are the coordinates of the perimeter

        if not np.all(phi_perimeter == 0):
            chance = phi_perimeter**eta
            chance[chance==1] = 0
            chance /= np.sum(chance)
            rand_idx = np.random.choice(perimeter.shape[0], p=chance)
        else:
            rand_idx = np.random.choice(perimeter.shape[0])

        cluster[i, :] = perimeter[rand_idx, :]
        perimeter = np.delete(perimeter, rand_idx, axis=0)
        candidates = cluster[i, :] + delta
        perimeter = np.append(perimeter, candidates[np.logical_and(~is_in_cluster(cluster, candidates), ~is_in_cluster(perimeter, candidates))], axis=0)

        #plt.imshow(phi)
        #plt.scatter(cluster[:,0], cluster[:,1], c='r', marker='x')
        #plt.scatter(perimeter[:,0], perimeter[:,1], c='b', marker='o')
        #plt.show()

    return cluster, perimeter, phi

def euc_dist_sq(x,y):
    """
    Calculates the square of the Euclidean norm of the vector [x, y]. A.K.A Pythagorean theorem. Calculating the square to avoid slowing down in the while loop by calculating square root every time.

    Parameters:
    ----------
    x : float
        the x Cartesian coordinate
    y : float
        the y Cartesian coordinate
    """
    d = x**2 + y**2
    return d

def Laplace(r_lead, cluster, BC_lead=1, BC_cluster=0, static=False):
    """
    Solves the Laplace equation given boundary conditions on a finite lattice.

    Parameters
    ----------
    r_lead : float
        the radius beyon which the potential is BC. Should be larger than the largest radial distance of a cluster point
    cluster : np.ndarray
        the position of all components of the cluster. Cluster is grounded and kept at zero potential
    BC : float
        boundary condition, the fixed potential value at the boundary

    Returns
    ----------
    phi : np.ndarray
        the electric potential over the entire lattice
    """
    lattice_size = int(r_lead)*2+1+2 #add one for origin of cluster, add two for space between edges and lead
    iterations = 1000#10*int(lattice_size)
    latt = np.indices((lattice_size, lattice_size))
    delta = int((lattice_size-1)/2) #shifting origin of cluster
    if static:
        cluster += delta

    kernel = generate_binary_structure(2,1)
    kernel[1,1]= False

    bool_beyond_lead = euc_dist_sq(latt[0,:,:]-delta, latt[1,:,:]-delta) >= r_lead**2

    phi = np.zeros((lattice_size, lattice_size))
    for i in range(1, iterations):
        #print('iter: ', i)
        phi = (1/4)*convolve(phi, kernel, mode='reflect') #each element is average of its neighbors
        phi[cluster[:,0], cluster[:,1]] = BC_cluster
        phi[bool_beyond_lead] = BC_lead

    return np.transpose(phi)

def is_in_cluster(cluster, points):
    """
    To check if candidate points are in the cluster or not

    Parameters
    ----------
    cluster : np.ndarray (number of points) x (2)
        coordinates of all the points in a cluster
    point : np.ndarray (n) x (2)
        coordinate of n points to check if in cluster

    Returns
    ----------
    is_in : list (1xn) (boolean)
        whether or not any point is in the cluster
    """
    is_in = np.array([np.any( np.all( (cluster == points[i, :]), axis=1) ) for i in range(points.shape[0])] )
    return is_in

def square(Nx, Ny):
    """
    Creates a coordinate array for a square lattice cenetered at 0

    Parameters
    ----------
    Nx : int (odd)
        the width of the square lattice. In order to be centered at 0 must be odd
    Ny : int (odd)
        the height of the square lattice. In order to be centered at 0 must be odd

    Returns
    ----------
    coor : np.ndarray (2)x(Nx*Ny)
        coordinate vectors for every lattice point
    """
    N = Nx*Ny
    coor = np.zeros((N,2))
    for i in range(Nx):
        for j in range(Ny):
            n = i + Nx*j
            coor[n, 0] = i - int((Nx-1)/2)
            coor[n, 1] = j - int((Ny-1)/2)
    return coor

def NN_Arr(coor):
    N = coor.shape[0]
    NN = -1*ones((N,4), dtype = 'int')
    xmax = max(coor[:, 0])
    ymax = max(coor[:, 1])
    Lx = int(xmax + 1)
    Ly = int(ymax + 1)

    for i in range(N):
        xi = coor[i, 0]
        yi = coor[i, 1]

        if (i-1) >= 0:
             if (xi - coor[i-1, 0]) == 1 and (yi - coor[i-1, 1]) == 0:
                 NN[i, 0] = i-1
        if (i+1) < N:
            if (xi - coor[i+1, 0]) == -1 and (yi - coor[i+1, 1]) == 0:
                NN[i, 2] = i+1
        for j in range(0, Lx+1):
            if (i+j) < N:
                 if (yi - coor[i+j, 1]) == -1 and (xi - coor[i+j, 0]) == 0:
                     NN[i, 1] = i+j
            if (i-j) >= 0:
                 if (yi - coor[i-j, 1]) == 1 and (xi - coor[i-j, 0]) == 0:
                     NN[i, 3]= i-j
    return NN

def circle(radius):
    """
    Creates an array of locations that belong to a disk with r = radius
    Used to determine density at certain radius

    Parameters
    ----------
    radius : int
        the radius of the disk
    step : int
        the width of the disk

    Returns
    ----------
    circle_pos : np.ndarray (2)x(..)
        coordinate vectors lattice points that belong to disk
    """
    circle_pos = np.empty(shape=[0, 2])
    minradius = radius-1/2
    maxradius = radius+1/2

    for x in range(radius+2):
        for y in range(radius+2):
            dist = np.sqrt(euc_dist_sq(x,y))
            if  minradius < dist and dist < maxradius:
                circle_pos = np.append(circle_pos, [[x, y]], axis=0)

    circle_pos1 = np.copy(circle_pos)
    circle_pos2 = np.copy(circle_pos)
    circle_pos3 = np.copy(circle_pos)
    circle_pos1[:,0] = -circle_pos1[:,0]
    circle_pos2[:,1] = -circle_pos2[:,1]
    circle_pos3 = -circle_pos3

    circle_pos = np.concatenate((circle_pos,circle_pos1,circle_pos2,circle_pos3),axis=0)

    circle_pos = np.unique(circle_pos, axis=0)

    return circle_pos

def func(x, a, b):
    """
    Function used for fitting the density density correlation (see func density_correlation)

    Parameters
    ----------
    x : np.ndarray 1D
        the radii
    a : float
        constant
    b : float
        constant

    Returns
    ----------
    y : np.ndarray 1D
        fitted density density correlation
    """

    y = a*x**b

    return y

def density(pos, N):
    """
    Calculates the density distribution as a function of r

    Parameters
    ----------
    pos : np.ndarray (2)x(..)
        the position of all components of the cluster
    step : float
        stepsize for calculating function (usually 1)
    N : integer
        number of walkers / components in cluster

    Returns
    ----------
    r : np.ndarray 1D
        radius vector

    dens : np.ndarray 1D
        density vector
    """

    rad = np.sqrt(euc_dist_sq(pos[:,0], pos[:,1]))
    r_max = int(rad.max()+1)
    r = np.linspace(0,r_max,r_max+1)
    dens = np.zeros(len(r))

    for i in range(len(r)):
        circ = circle(int(r[i]))
        aset = set([tuple(x) for x in pos])
        bset = set([tuple(x) for x in circ])
        intersection = np.array([x for x in aset & bset])

        dens[i] = len(intersection)/ len(circ)

    return r, dens

def density_correlation(r, dens, N):

    """
    Calculates the density-density correlation function (to be able to retrieve Hausdorf dimension)

    Parameters
    ----------
    dens : np.ndarray 1D
        the densities at r
    r : np.ndarray 1D
        radius from center
    N : integer
        number of walkers / components in cluster

    Returns
    ----------
    cor_dens : np.ndarray 1D
        density density correlation vector
    alpha : np.ndarray 1D 1x2
        fitted parameters
    std : np.ndarray 1D 1x2
        std of the fitted parameters
    """

    r_max = np.amax(r)
    cor_dens = np.zeros(len(r))

    for i in range(len(r)):
        for j in range(len(r)-i):
            cor_dens[i] += dens[j+i]*dens[j]

    cor_dens = cor_dens/N

    min_r_fit = 5
    max_r_fit = int(r_max*0.5)
    alpha, beta = optimize.curve_fit(func, xdata = r[min_r_fit :max_r_fit], ydata = cor_dens[min_r_fit:max_r_fit]) #alpha contains fitted par in func, beta is cov matrix
    std = np.sqrt(np.diag(beta)) #derive std of par

    return cor_dens, alpha, std

def fractal_dim(cluster):
    rad = np.sqrt(euc_dist_sq(cluster[:,0], cluster[:,1]))
    r_max = int(rad.max()+1)
    r = np.linspace(0,r_max,r_max+1)
    dens = np.zeros(len(r))

def c(cluster, r):
    """
    Measures the number of cluster particles within a radius r

    Parameters
    ----------
    cluster : np.ndarray (2) x (N)
        the cluster being used
    r : float
        the radius of the circle to add particles within

    returns
    ----------
    num_within_circle : int
        the number of cluster particles within a circle of radius r from the origin
    """
    num_part = np.sqrt(euc_dist_sq(cluster[:,0], cluster[:,1]))<r
    num_part = num_part[num_part]
    num_within_circle = num_part.shape[0]
    return num_within_circle
