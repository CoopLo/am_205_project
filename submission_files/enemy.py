import numpy as np
import os
from numba import jit
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def kinematics_acc(vx0, vy0, vz0, ax, ay, az, t, x0=0, y0=0, z0=0, g=0, vt=9999999999):
    # from http://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node29.html
    # Simple kinematics
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
    velocity = np.linalg.norm([vx, vy, vz])
    vx += vx_sched*delta_t/mass_tot*(-1/2*rho_0*np.exp(z/8000)*velocity**2*A*C_d - c*delta_m)
    vy += vy_sched*delta_t/mass_tot*(-1/2*rho_0*np.exp(z/8000)*velocity**2*A*C_d - c*delta_m)
    vz += vz_sched*delta_t/mass_tot*(-mass_tot*g - 1/2*rho_0*np.exp(z/8000)*velocity**2*A*C_d - \
                                     c*delta_m)

    # Update positions
    x += vx*delta_t
    y += vy*delta_t
    z += vz*delta_t

    # Update mass
    r_mass += delta_m
    return x, y, z, vx, vy, vz, r_mass


def write(x, y, z, vx, vy, vz):
    # Write final data
    with open("./parallel_test/enemy_x.txt", 'a+') as fout:
        fout.write("{}\n".format(x))
    fout.close()
    with open("./parallel_test/enemy_y.txt", 'a+') as fout:
        fout.write("{}\n".format(y))
    fout.close()
    with open("./parallel_test/enemy_z.txt", 'a+') as fout:
        fout.write("{}\n".format(z))
    fout.close()
    with open("./parallel_test/enemy_vx.txt", 'a+') as fout:
        fout.write("{}\n".format(vx))
    fout.close()
    with open("./parallel_test/enemy_vy.txt", 'a+') as fout:
        fout.write("{}\n".format(vy))
    fout.close()
    with open("./parallel_test/enemy_vz.txt", 'a+') as fout:
        fout.write("{}\n".format(vz))
    fout.close()


def rocket(delta_t=0.025):
    # Simulation variables
    steps = int(1000 * 0.025/delta_t)

    # Enemy system variables
    vx = 0
    vy = 0 
    vz = 0
    xs = [0]
    ys = [0]
    zs = [0]

    vxs = [vx]
    vys = [vy]
    vzs = [vz]
    y = 0
    x = 0
    z = 0

    C_d = 0.1                            # Coefficient of drag
    c = 70000                            # Exhaust force
    delta_m = -1*(delta_t/0.025)         # Change in mass
    A = 0.25                             # Cross sectional area
    mass = 1000                          # Total mass of rocket without fuel
    r_mass = 1000                        # Mass of fuel
    rho_0 = 1.2754                       # Initial density of air

    #print(delta_m)
    vx_sched = np.ones(int(-mass/delta_m))*.45
    vy_sched = np.ones(int(-mass/delta_m))*.45
    vz_sched = np.ones(int(-mass/delta_m))*0.1

    # Changing Flight path
    vz_sched[int(200 * (0.025/delta_t)): int(400 * (0.025/delta_t))] *= -1

    vx_sched[int(400 * (0.025/delta_t)):int(500 * (0.025/delta_t))] += 0.15
    vy_sched[int(400 * (0.025/delta_t)):int(500 * (0.025/delta_t))] -= 0.05
    vz_sched[int(400 * (0.025/delta_t)):int(500 * (0.025/delta_t))] *= 0

    vx_sched[int(500 * (0.025/delta_t)):] -= 0
    vy_sched[int(500 * (0.025/delta_t)):] -= 0
    vz_sched[int(500 * (0.025/delta_t)):] *= -1


    # Constants
    g = 9.81
    vt = g*(mass)/C_d  # Terminal velocity only shows up in kinematics
    #print("TERMINAL VELOCITY: {}".format(vt))
    j = 0
    done = False
    for i in tqdm(range(steps)):
        if(r_mass > 0):
            x, y, z, vx, vy, vz, r_mass = rocket_equation(vx, vy, vz, i*delta_t, x, y, z,
                                            g, rho_0, mass, r_mass, delta_t, delta_m, A, C_d, c,
                                            vx_sched[i], vy_sched[i], vz_sched[i])

        elif(not done): # Initial parameters for switching from rocket to kinematics
            print("KINEMATICS AT STEP: {}".format(i))
            init_x = x
            init_y = y
            init_z = z
            vx0 = vx
            vy0 = vy
            vz0 = vz
            done = True
            kt = i # Timestep we switched

        if(done):
            x, y, z, vx, vy, vz = kinematics(vx0, vy0, vz0, (i-kt)*delta_t,
                                             init_x, init_y, init_z, g, vt)

        if(z < 1e-2 and i > 100):
            break

        # Hold on to current values
        write(x, y, z, vx, vy, vz)
        time.sleep(delta_t)


if __name__ == '__main__':
    rocket()

