import numpy as np
import os
from numba import jit
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot(steps, i, x, y, z, x_up, y_up, z_up, kt):
    color = 'r' if(i <= kt) else 'b'
    label = "THRUST" if(i <= kt) else "NO THRUST"

    leading_zeros = int(np.log(steps)/np.log(10))
    leading_zeros -= 0 if(i==0) else int(np.log(i)/np.log(10))
    zeros = '0'*leading_zeros
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color=color, label=label)
    ax.set(xlim=(-0.5, x_up), ylim=(-0.5, y_up), zlim=(-0.5, z_up),
           xlabel="x", ylabel="y", zlabel="z")
    ax.legend(loc='best')
    plt.savefig("./graph_missile/{}{}.png".format(zeros,i))
    plt.close(fig=fig)


@jit
def kinematics(vx0, vy0, vz0, t, x0=0, y0=0, z0=0, g=0, vt=0):
    # from http://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node29.html
    vx = vx0 + vx0*np.exp(-g*t/vt)
    vy = vy0 + vy0*np.exp(-g*t/vt)
    vz = vz0 + vz0*np.exp(-g*t/vt) - vt*(1- np.exp(-g*t/vt)) # Height dependent drag

    x = x0 + vx0*vt/g*(1 - np.exp(-g*t/vt))
    y = y0 + vy0*vt/g*(1 - np.exp(-g*t/vt))
    z = z0 + vt/g*(vz0 + vt)*(1 - np.exp(-g*t/vt)) - vt*t
    return x, y, z, vx, vy, vz


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
    print(z)
    return x, y, z, vx, vy, vz, r_mass


def rocket():
    # Simulation variables
    steps = 1000
    delta_t = 0.025

    # System variables
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


    C_d = 0.1            # Coefficient of drag
    c = 50000             # Exhaust force
    delta_m = -1         # Change in mass
    A = 0.25              # Cross sectional area
    mass = 1000           # Total mass of rocket without fuel
    r_mass = 1000         # Mass of fuel
    rho_0 = 1.2754        # Initial density of air

    vx_sched = np.ones(int(-mass/delta_m))*.45
    vy_sched = np.ones(int(-mass/delta_m))*.45
    vz_sched = np.ones(int(-mass/delta_m))*0.1

    # Changing Flight path
    vz_sched[200:400] *= -1

    vx_sched[400:500] += 0.05
    vy_sched[400:500] += 0.05
    vz_sched[400:500] *= 0

    vx_sched[500:] -= 0.1
    vy_sched[500:] -= 0.1
    vz_sched[500:] *= -2


    # Constants
    g = 9.81
    vt = g*(mass)/C_d  # Terminal velocity only shows up in kinematics
    print("TERMINAL VELOCITY: {}".format(vt))
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
            #if(i < 65):
            #    print("INIT_X: {}".format(init_x))
            #    print("INIT_Y: {}".format(init_y))
            #    print("INIT_Z: {}".format(init_z))
            #    print("X: {}".format(x))
            #    print("Y: {}".format(y))
            #    print("Z: {}".format(z))
            #    print("VX0: {}".format(vx0))
            #    print("VY0: {}".format(vy0))
            #    print("VZ0: {}".format(vz0))
            #    print("VX: {}".format(vx))
            #    print("VY: {}".format(vy))
            #    print("VZ: {}".format(vz))
            #    print("\n\n")
        xs.append(x)
        ys.append(y)
        zs.append(z)
        vxs.append(vx)
        vys.append(vy)
        vzs.append(vz)
        if(z < 1e-2 and i>100):
            break

    end = True
    maxx = []
    maxy = []
    maxz = []
    fig, ax = plt.subplots()
    #ax.plot(vxs)
    #ax.plot(vys)
    #ax.plot(vzs)
    #plt.show()
    #exit(1)

    for i in range(len(xs)):
        maxx.append(np.max(xs[i]))
        maxy.append(np.max(ys[i]))
        maxz.append(np.max(zs[i]))

    kt = 1000
    for i in tqdm(range(0, len(xs))):
        plot(steps, i, xs[i], ys[i], zs[i],
                   1.1*np.max(maxx), 1.1*np.max(maxy), 1.1*np.max(maxz), kt)

    print("WRITING FILE")
    np.savetxt("./graph_missile/x_dat.txt", xs)
    np.savetxt("./graph_missile/y_dat.txt", ys)
    np.savetxt("./graph_missile/z_dat.txt", zs)
    np.savetxt("./graph_missile/x_vel.txt", vxs)
    np.savetxt("./graph_missile/y_vel.txt", vys)
    np.savetxt("./graph_missile/z_vel.txt", vzs)
    fig, ax = plt.subplots()
    ax.plot(vxs, label="X VELOCITY")
    ax.plot(vys, label="Y VELOCITY")
    ax.plot(vzs, label="Z VELOCITY")
    ax.legend(loc='best')
    plt.savefig("./graph_missile/velocities.png")
    os.system("cd ./graph_missile/ ; ffmpeg -pattern_type glob -i \"*.png\" -c:v libx264 -pix_fmt yuv420p -movflags +faststart output.mp4")
    #plt.show()


if __name__ == '__main__':
    rocket()

