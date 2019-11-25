import numpy as np
import os
from numba import jit
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def inside_hemisphere(x, y, z, center, radius):
    return np.sqrt((x-center[0])**2 + (y-center[1])**2 + z**2) < radius

def plot(steps, i, x, y, z, x_up, y_up, z_up, kt, center, radius, inside):
    color = 'r' if(i <= kt) else 'b'
    label = "THRUST" if(i <= kt) else "NO THRUST"
    dome_color = 'g' if(inside) else 'b'

    leading_zeros = int(np.log(steps)/np.log(10))
    leading_zeros -= 0 if(i==0) else int(np.log(i)/np.log(10))
    zeros = '0'*leading_zeros
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot rocket
    ax.scatter(x, y, z, color=color, label=label)

    # Plot detection hemisphere
    r = 500
    phi, theta = np.mgrid[0.0:0.5*np.pi:180j, 0.0:2.0*np.pi:720j] # phi = alti, theta = azi
    x = radius*np.sin(phi)*np.cos(theta) + center[0]
    y = radius*np.sin(phi)*np.sin(theta) + center[1]
    z = radius*np.cos(phi)
    ax.plot_surface(x, y, z, alpha=0.3, color=dome_color)
    ax.scatter(*center, color='k')

    ax.set(xlim=(-0.5, x_up), ylim=(-0.5, y_up), zlim=(-0.5, 500), #zlim=(-0.5, z_up),
           xlabel="x", ylabel="y", zlabel="z")
    ax.legend(loc='best')
    plt.savefig("./track_graph_missile/{}{}.png".format(zeros,i))
    plt.close(fig=fig)
    #exit(1)


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
    return x, y, z, vx, vy, vz, r_mass


@jit
def response():
    pass

def rocket():
    # Simulation variables
    steps = 1000
    delta_t = 0.025

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


    # Intercepting system variables
    center = [1000, 1000]
    radius = 500
    inside = []
    vx = 0
    vy = 0
    vz = 0
    xs = [center[0]]
    ys = [center[1]]
    zs = [0]
    target_x = []
    target_y = []
    target_z = []
    evx = []
    evy = []
    evz = []
    estimation = []

    C_d = 0.1            # Coefficient of drag
    r_c = 300000           # Exhaust force
    r_delta_m = -1         # Change in mass
    r_A = 0.25              # Cross sectional area
    r_mass = 1000           # Total mass of rocket without fuel
    r_r_mass = 1000         # Mass of fuel


    rho_0 = 1.2754        # Initial density of air
    response = False


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

        inside.append(inside_hemisphere(x, y, z, center, radius))

        # Enemy missile has been detected
        if(any(inside)):
            response = True

        # Send response missile
        if(response):
            # LAUNCH RESPONSE MISSILE
            # AIMS A FEW TIMESTEPS IN THE FUTURE ACCORDING TO LINEAR EXTRAPOLATION?
            # NEED 3 DATAPOINTS FOR ACCELERATION
            target_x.append(x)
            target_y.append(y)
            target_z.append(z)

            # Estimate new velocity and acceleration for enemy missile
            if(len(target_x) > 3):
                estimation.append(i)
                target_x = target_x[1:]
                target_y = target_y[1:]
                target_z = target_z[1:]

                x_vel = (target_x[-1] - target_x[-2])/delta_t
                y_vel = (target_y[-1] - target_y[-2])/delta_t
                z_vel = (target_z[-1] - target_z[-2])/delta_t

                x_acc = (target_x[0] - 2*target_x[1] + target_x[2])/(delta_t**2)
                y_acc = (target_y[0] - 2*target_y[1] + target_y[2])/(delta_t**2)
                z_acc = (target_z[0] - 2*target_z[1] + target_z[2])/(delta_t**2)

                evx.append(x_vel)
                evy.append(y_vel)
                evz.append(z_vel)
                
                # MAYBE TARGET POINT CLOSER AND CLOSER TO ENEMY AS WE APPROACH?
                # ASSUMING ALL PERFECT KNOWLEDGE OF EVERYTHING


        # Hold on to current values
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
    #for i in tqdm(range(0, len(xs))):
    #    plot(steps, i, xs[i], ys[i], zs[i],
    #         1.1*np.max(maxx), 1.1*np.max(maxy), 1.1*np.max(maxz), kt,
    #         center, radius, inside[i])

    print("WRITING FILE")
    np.savetxt("./track_graph_missile/x_dat.txt", xs)
    np.savetxt("./track_graph_missile/y_dat.txt", ys)
    np.savetxt("./track_graph_missile/z_dat.txt", zs)
    np.savetxt("./track_graph_missile/x_vel.txt", vxs)
    np.savetxt("./track_graph_missile/y_vel.txt", vys)
    np.savetxt("./track_graph_missile/z_vel.txt", vzs)
    fig, ax = plt.subplots()
    ax.plot(vxs, label="X VELOCITY")
    ax.plot(vys, label="Y VELOCITY")
    ax.plot(vzs, label="Z VELOCITY")

    ax.plot(estimation, evx, label="ESTIMATED X VELOCITY")
    ax.plot(estimation, evy, label="ESTIMATED Y VELOCITY")
    ax.plot(estimation, evz, label="ESTIMATED Z VELOCITY")

    ax.legend(loc='best')
    plt.savefig("./track_graph_missile/velocities.png")
    #os.system("cd ./track_graph_missile/ ; ffmpeg -pattern_type glob -i \"*.png\" -c:v libx264 -pix_fmt yuv420p -movflags +faststart output.mp4")
    #plt.show()


if __name__ == '__main__':
    rocket()

