#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTE 404 Project

This script prompts user for NACA airfoil characteristics and plots the 
resulting airfoil.
"""

import numpy as np
import math
import matplotlib.pyplot as plt

print("\n The NACA four-digit wing sections define the profile by: \n")
print("(1) First digit describing maximum camber as percentage of the chord.")
print("(2) Second digit describing the distance of maximum camber from the")
print("\t airfoil leading edge in tenths of the chord.")
print("(3) Last two digits describing maximum thickness of the airfoil as")
print("\t percent of the chord.\n")

M = float(input("\n Enter Max Camber (%), 0 - 9.5%: ")) / 100
P = float(input("\n Enter Max Camber Position (%), 0 - 90%: ")) / 100
T = float(input("\n Enter Thickness (%), 1 - 40%: ")) / 100
numPoints = int(input("\n Enter Number of Points, 20 to 200: "))

# Evenly spaced array of numPoints elements from 0 to pi
# Using cosine spacing to concentrate extra elements near 0 and 1 
beta = np.linspace(0, math.pi, num=numPoints, endpoint=True)
x = (1-np.cos(beta)) / 2

# Camber equations
if P != 0:
    yc_front = (M/P**2) * ( 2*P*x[x<P] - np.power(x[x<P],2) )
else:
    yc_front = np.array([])
yc_back = (M/(1-P)**2) * ( 1 - 2*P + 2*P*x[x>=P] - np.power(x[x>=P],2) )
yc = np.concatenate((yc_front, yc_back))

# Gradient equations
if P != 0:
    dycdx_front = (2*M/P**2) * (P-x[x<P])
else:
    dycdx_front = np.array([])
dycdx_back = (2*M/(1-P)**2) * (P-x[x>=P])
dycdx = np.concatenate((dycdx_front, dycdx_back))

# Thickness distribution
a0 = 0.2969
a1 = -0.126
a2 = -0.3516
a3 = 0.2843
a4 = -0.1036 # closed trailing edge; use -0.1015 for open trailing edge

yt = (T/0.2) * ( a0*np.power(x,0.5) + a1*x + a2*np.power(x,2) \
                + a3*np.power(x,3) + a4*np.power(x,4) )

# Calculate envelope positions perpendicular to camber line  
theta = np.arctan(dycdx)

# Upper line
xu = x - yt*np.sin(theta)
yu = yc + yt*np.cos(theta)

# Lower line
xl = x + yt*np.sin(theta)
yl = yc - yt*np.cos(theta)

# Plot
fig, ax = plt.subplots()
ax.plot(xl,yl)
ax.plot(xu,yu)
ax.set_ylim([-0.5,0.5])

    
