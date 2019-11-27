import numpy as np
import os
from numba import jit
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@jit
def kinematics(vx0, vy0, vz0, t, x0=0, y0=0, z0=0, g=0, vt=0):
    # from http://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node29.html
    vx = vx0 + vx0*np.exp(-g*t/vt)
    vy = vy0 + vy0*np.exp(-g*t/vt)
    vz = vz0 + vz0*np.exp(-g*t/vt) - vt*(1- np.exp(-g*t/vt)) # NEED Height dependent drag

    x = x0 + vx0*vt/g*(1 - np.exp(-g*t/vt))
    y = y0 + vy0*vt/g*(1 - np.exp(-g*t/vt))
    z = z0 + vt/g*(vz0 + vt)*(1 - np.exp(-g*t/vt)) - vt*t
    return x, y, z, vx, vy, vz


def kinematics_acc(vx0, vy0, vz0, ax, ay, az, t, x0=0, y0=0, z0=0, g=0, vt=0):
    x = x0 + vx0*vt/g*(1 - np.exp(-g*t/vt))
    y = y0 + vy0*vt/g*(1 - np.exp(-g*t/vt))
    z = z0 + vt/g*(vz0 + vt)*(1 - np.exp(-g*t/vt)) - vt*t
    return x, y, z


@jit
def rocket_equation(vx, vy, vz, t, x, y, z, g, rho_0, mass, r_mass, delta_t, delta_m, A, C_d, c,
                    vx_sched, vy_sched, vz_sched):
    # Set up total mass
    mass_tot = mass + r_mass

    # Update velocities
    vx += vx_sched*delta_t/mass_tot*(-1/2*rho_0*np.exp(z/8000)*vx**2*A*C_d - c*delta_m)
    vy += vy_sched*delta_t/mass_tot*(-1/2*rho_0*np.exp(z/8000)*vy**2*A*C_d - c*delta_m)
    vz += vz_sched*delta_t/mass_tot*(-mass_tot*g - 1/2*rho_0*np.exp(z/8000)*vz**2*A*C_d - \
                                     c*delta_m)

    # Update positions
    x += vx*delta_t
    y += vy*delta_t
    z += vz*delta_t

    # Update mass
    r_mass += delta_m
    return x, y, z, vx, vy, vz, r_mass


def respones(measurements):
    # Simulation variables
    steps = 1000
    delta_t = 0.025

    # Intercepting system variables
    center = [1000, 1000]
    radius = 500
    inside = [False]

    # Initital position
    xs = [center[0]]
    ys = [center[1]]
    zs = [0]

    # Initial velocity
    vxs = [0]
    vys = [0]
    vzs = [0]

    # Target position
    target_x = []
    target_y = []
    target_z = []

    proj_x = [center[0]]
    proj_y = [center[1]]
    proj_z = [0]
    cur_dists = []

    C_d = 0.1            # Coefficient of drag
    c = 100000           # Exhaust force
    projection = 115
    time_projection = 115
    delta_m = -1         # Change in mass
    A = 0.25             # Cross sectional area
    mass = 1000          # Total mass of rocket without fuel
    r_mass = 1000        # Mass of fuel
    current_dist = 500


    rho_0 = 1.2754        # Initial density of air
    response = False


    # Constants
    g = 9.81
    vt = g*(mass)/C_d  # Terminal velocity only shows up in kinematics
    j = 0
    done = False
    
    # Need to estimate...
    # vx, vy, vz, ax, ay, ax, drag force (area, drag coefficient), change in mass
    # Project out enemy missile
    for i in range(1000):
        pass

    # Response missile
    for i in tqdm(range(steps)):
        x, y, z, vx, vy, vz, r_mass = rocket_equation(vxs[-1], vys[-1], vzs[-1], i*delta_t,
                                       xs[-1], ys[-1], zs[-1],
                                       g, rho_0, mass, r_mass, delta_t, delta_m, A, C_d, c,
                                       vx_sched[i], vy_sched[i], vz_sched[i])

        xs.append(x)
        ys.append(y)
        zs.append(z)
        vxs.append(vx)
        vys.append(vy)
        vzs.append(vz)



if __name__ == '__main__':
    rocket()

