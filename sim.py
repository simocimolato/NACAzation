#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 19:05:31 2023

@author: simonecimolato

NACAZATION

Using Lattice - Boltzmann D2Q9 scheme
"""

import numpy as np
import matplotlib.pyplot as plt

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def curl(ux, uy):
    dfydx = ux[2:, 1:-1] - ux[0:-2, 1:-1]
    dfxdy = uy[1:-1, 2:] - uy[1:-1, 0:-2]

    return dfydx - dfxdy

plot_every = 50

#def sim():
Nx = 200        # number of lattices in the X dimension (columns)
Ny = 75        # number of lattices in the Y dimension (rows)

tau = 0.53      # kinematic viscosity (or time scale) for tau < 0.52 the simulation is not stable
rndmK = 0.01    # magnitude of random flactuations in initial speed of the fluid
Nt = 30000      # number of time iterations

NL = 9          # number of velocity directions allowed

# lattice speeds
cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])

# lattice weights
w = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

# initial conditions
F = np.ones((Ny, Nx, NL)) + rndmK * np.random.randn(Ny, Nx, NL)
F[:, :, 3] = 2          # initial rightward velocity (momentum is constant, so it will have a permanent rightward velocity)

# collisions
Feq = np.zeros(F.shape)

""" creating the airfoil"""
# location in the matrix (using //, the floor division, as coordinates must be int)
xAir = Nx // 4
yAir = Ny // 2

foil = np.full((Ny, Nx), False)

for x in range (0, Nx) :
    for y in range(0, Ny):
            if(distance (xAir, yAir, x, y) < 10):
                foil[y][x] = True

"""main loop"""
for t in range(Nt):
    # left and right wall absorbant boundary condition
    F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]
    F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]

    # top and bottom walls
    F[0, :, [4, 5, 6]] = F[1, :, [4, 5, 6]]
    F[-1, :, [8, 1, 2]] = F[-2, :, [8, 1, 2]]

    # stream
    for i, cx, cy in zip(range(1,NL), cxs[1:], cys[1:]):
        F[:, :, i] = np.roll(F[:, :, i], cx, axis = 1)
        F[:, :, i] = np.roll(F[:, :, i], cy, axis = 0)
        #F[:, ;, i] = np.concatenate(([0], arr[:-1]))
        # concatenate instead of roll so that the velocities at the end dont get put in the front again

    # boundaries
    bndryF = F[foil, :]
    bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

    rho = np.sum(F, 2)      # densisty as sum of velocities for every lattice

    ux = np.sum(F * cxs, 2) / rho       # momentum along the x-axis
    uy = np.sum(F * cys, 2) / rho       # momentum along the y-axis

    # setting velocity inside the body at 0
    F[foil, :] = bndryF
    ux[foil] = 0
    uy[foil] = 0

    # collisions
    for i, cx, cy, k in zip(range(NL), cxs, cys, w):
        Feq[:, :, i] = rho * k * (1 + 3 * (cx * ux + cy * uy) + (9/2) *((cx * ux + cy * uy) ** 2) - (3/2) * (ux ** 2 + uy ** 2))

    F = F - (1/tau) * (F - Feq)

    # plots
    if t % plot_every == 0:
        velocity = np.sqrt(ux ** 2 + uy ** 2)

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Set titles for the subplots
        ax1.set_title('Velocity')
        ax2.set_title('Curl')

        # plot
        im1 = ax1.imshow(velocity, cmap='plasma')
        im2 = ax2.imshow(curl(ux, uy), cmap='bwr')

        plt.pause(0.001)
        plt.close()     # do we need to close the fig and create it again everytime?


"""
TODO:
    make it faster by possibly not checking every lattice point at every timestep
    do we need to know and calculate the 0th velocity?
    make it faster by changin absorbant walls method
    make it faster
    airfoil with the x,y points,
    figure out plot_every and time iterations, it's too hard coded
    making a funcion out of this and putting it in a program
    cd and cl calculation? pressure diff? other stuff?
    do we need to close the fig and create it again everytime?
    make it robust to use cases and ways of using it
    neural net to predict all this shit
"""