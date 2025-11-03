# Fractal clusters

Simulation of Diffusion-Limited Aggregation (DLA) and Eden-type growth models to generate fractal clusters.

## Overview

Diffusion-limited aggregation (DLA) describes the stochastic formation of fractal clusters by particles undergoing random Brownian motion and sticking to a seed or existing aggregate. Such models are relevant to bacterial/tumor growth, snowflake formation, and molecular beam epitaxy.  

This project constructs DLA clusters in different environments and extracts structural properties such as the fractal (Hausdorff) dimension, diffusion constant, and correlation functions. Results are compared against known literature.

![DLA_rainbow](examples/gifs/DLA_cluster__animation_1000.gif)

The Eden model is another stochastic growth model used to describe dielectric breakdown, bacterial proliferation, and tumor interface dynamics. 

![Eden_rainbow](examples/gifs/Eden_cluster_animation_950.gif)

In the dielectric breakdown model (DBM), the local electric potential influences the probability distribution for where the cluster grows.

## Literature

- [Witten, Sander. (1981). "Diffusion-Limited Aggregation, a Kinetic Critical Phenomenon". Physical Review Letters. 47 (19). doi: 10.1103/PhysRevLett.47.1400.](https://doi.org/10.1103/PhysRevLett.47.1400)
- [Witten, Sander. (1983). "Diffusion-Limited Aggregation". Physical Review B. 27 (9). doi: 10.1103/PhysRevB.27.5686.](https://doi.org/10.1103/PhysRevB.27.5686)
- [Niemeyer, Pietronero, Wiesmann. (1984). "Fractal Dimension of Dielectric Breakdown". Physical Review Letters. 52 (12). doi: 10.1103/PhysRevLett.52.1033]( https://doi.org/10.1103/PhysRevLett.52.1033)
  
## Relevant references

- https://en.wikipedia.org/wiki/Diffusion-limited_aggregation
- https://www.cmt.york.ac.uk/compmag/resources/2.2A.pdf
- https://www.astro.rug.nl/~offringa/Diffusion%20Limited%20Aggregation.pdf
- https://www.ippp.dur.ac.uk/~krauss/Lectures/NumericalMethods/Percolation/Lecture/pe2.html
- https://linuxtut.com/en/e4b80611a562480cb1af/
- https://users.math.yale.edu/public_html/People/frame/Fractals/Panorama/Physics/DLA/DLA6.html

## Instructions

The user of this repository can access multiple files that show results form our two models. 

> Note: Simulation files ([dielec_brkdwn_sim.py](dielec_brkdwn_sim.py), [dla_sim.py](dla_sim.py), [eden_sim.py](eden_sim.py), [static_potential_sim.py](static_potential_sim.py)) are used to both write data to ([\dielectric_data](dielectric_data), [\dla_data](dla_data), [\eden_data](eden_data)) and save figures to [\figs](figs). 
> - To create new data or overwrite prexisting data, run the simulation files as normal `python file_sim.py`
> - To use existing data to plot figures use the command line prompt as above with a 'P' added to the end `python file_sim.py P`. This will bypass the code running the simulation and load the data with the parameters specified at the beginning of the file.  



## File descriptions 

### [fractal.py](fractal.py) 
The file containing all usable functions

### [dla_sim.py](dla_sim.py)
DLA simulation to construct a cluster using different numbers of random walkers through the parameter `nr_walkers`. The parameter `n` is a string that can characterize different simulations of the same number of particles, so as to avoid overwriting previous simulations. To plot figures with different color properties, can change one of the boolean values `rainbow`, `colorful`, `blue` to `True` and run the file as described in the instructions. 

### [eden_sim.py](eden_sim.py)
Eden cluster growth simulation using different number of cells in the cluster through a parameter `nr_cells`. 

### [dielec_brkdwn_sim.py](dielec_brkdwn_sim.py)
Eden growth simulation in an electric potetntial. Can change the number of cells in the cluster through `nr_cells`. Can change the strength of the effect of the local electric potential on the probability of growth through `eta`. Can change the boundary conditions on the lead and cluster through `BC_lead` and `BC_cluster` respectively

### [static_potential_sim.py](static_potential_sim.py)
Can visualize the electric potential formed by a circular lead at fixed potential `BC_lead` at radius `r_lead` and an already formed cluster which is set to potential `BC_cluster`. Importantly, the growth of the cluster is produced separate from the potential environment.

### [dla_density_cor.py](dla_density_cor.py)
Measures the density correlation function through the average of several grown DLA clusters with 10,000 aggregate particles. Performs a best fit in order to extract numerical fractal quantities. 

#### [fractal_dim.py](fractal_dim.py)
Measures the fractal dimension (Hausdorff dimension) based of the average number of particles within a given radius of four DLA clusters with 10,000 random walkers. 

### [walker_ani.py](walker_ani.py)
Animates a random walkers path

### [profiling.py](profiling.py)
Measures the performance of several functions from [fractal.py](fractal.py)
