import numpy as np
import os
from numba import jit
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def inside_hemisphere(x, y, z, center, radius):
    return np.sqrt((x-center[0])**2 + (y-center[1])**2 + z**2) < radius

def plot(steps, i, x, y, z, x_up, y_up, z_up, kt, center, radius, inside,
         proj_x, proj_y, proj_z, dome_x, dome_y, dome_z, r_x, r_y, r_z, done=False):
    color = 'r' if(i <= kt) else 'b'
    label = "ENEMY" if(i <= kt) else "NO THRUST"
    dome_color = 'g' if(inside) else 'b'

    leading_zeros = int(np.log(steps)/np.log(10))
    leading_zeros -= 0 if(i==0) else int(np.log(i)/np.log(10))
    zeros = '0'*leading_zeros
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot rocket
    s = 0 if(done) else 20
    ax.scatter(x, y, z, color=color, label=label, s=s)

    # Plot detection hemisphere
    ax.plot_surface(dome_x, dome_y, dome_z, alpha=0.3, color=dome_color)
    ax.scatter(*center, color='k', label="SENSOR")

    # Plot position projection
    ax.scatter(proj_x, proj_y, proj_z, color='k', marker='x', label="POSITION PROJECTION")

    # Plot response rocket
    s = 100 if(done) else 20
    label = "SUCCESSFUL INTERCEPTION" if(done) else "RESPONSE"
    ax.scatter(r_x, r_y, r_z, color='g', label=label, s=s)

    x_up = max(x_up, center[0]+radius)
    y_up = max(y_up, center[0]+radius)
    ax.set(xlim=(-0.5, x_up), ylim=(-0.5, y_up), zlim=(-0.5, z_up), #zlim=(-0.5, 500),
           xlabel="x", ylabel="y", zlabel="z")
    ax.legend(loc='upper left')
    plt.savefig("./track_graph_missile/{}{}.png".format(zeros,i))
    plt.close(fig=fig)
    #exit(1)


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
    c = 70000             # Exhaust force
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
    #time_projection = 115
    center = [1000, 1000]
    radius = 500
    inside = [False]
    r_vx = 0
    r_vy = 0
    r_vz = 0
    r_x = center[0]
    r_y = center[1]
    r_z = 0
    r_xs = [center[0]]
    r_ys = [center[1]]
    r_zs = [0]
    r_vxs = [0]
    r_vys = [0]
    r_vzs = [0]
    evx = []
    evy = []
    evz = []
    target_x = []
    target_y = []
    target_z = []
    estimation = []
    proj_x = [center[0]]
    proj_y = [center[1]]
    proj_z = [0]
    cur_dists = []

    r_C_d = 0.1            # Coefficient of drag
    r_c = 100000           # Exhaust force
    projection = 115
    time_projection = 115
    r_delta_m = -1         # Change in mass
    r_A = 0.25             # Cross sectional area
    r_mass = 1000          # Total mass of rocket without fuel
    r_r_mass = 1000        # Mass of fuel
    current_dist = 500


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
                # Adjust estimation parameters
                estimation.append(i)
                target_x = target_x[1:]
                target_y = target_y[1:]
                target_z = target_z[1:]

                # Calculate new x velocities
                x_vel = (target_x[-1] - target_x[-2])/delta_t
                y_vel = (target_y[-1] - target_y[-2])/delta_t
                z_vel = (target_z[-1] - target_z[-2])/delta_t

                # Calculate new accelerations
                x_acc = (target_x[0] - 2*target_x[1] + target_x[2])/(delta_t**2)
                y_acc = (target_y[0] - 2*target_y[1] + target_y[2])/(delta_t**2)
                z_acc = (target_z[0] - 2*target_z[1] + target_z[2])/(delta_t**2)

                evx.append(x_vel)
                evy.append(y_vel)
                evz.append(z_vel)

                # Project target point
                px, py, pz, = kinematics_acc(x_vel, y_vel, z_vel, x_acc, y_acc, z_acc,
                                    time_projection*delta_t,
                                    target_x[-1], target_y[-1], target_z[-1], g, vt)
                
                proj_x.append(px)
                proj_y.append(py)
                proj_z.append(pz)

                # Aim response missile at target point
                cur_dists.append(current_dist)

                dist_vec = [(px-r_x), (py - r_y), (pz - r_z)]
                #dist_vec = [(target_x[-1]-r_x), (target_y[-1] - r_y), (target_z[-1] - r_z)]
                #print(dist_vec)
                azimuth = np.arccos(dist_vec[2]/np.linalg.norm(dist_vec))
                radial = np.arctan(dist_vec[1]/dist_vec[0])

                x_frac = np.sin(azimuth)*np.cos(radial)
                y_frac = np.sin(azimuth)*np.sin(radial)
                z_frac = np.cos(azimuth)
                normalize = x_frac + y_frac + z_frac

                x_frac /= normalize
                y_frac /= normalize
                z_frac /= normalize

                x_frac = dist_vec[0]/np.linalg.norm(dist_vec)
                y_frac = dist_vec[1]/np.linalg.norm(dist_vec)
                z_frac = dist_vec[2]/np.linalg.norm(dist_vec)

                #print("X: {}, Y: {}, Z: {}\n\n".format(x_frac, y_frac, z_frac))

                r_x, r_y, r_z, r_vx, r_vy, r_vz, r_r_mass = rocket_equation(r_vx, r_vy, r_vz, 
                                            (i-estimation[0])*delta_t,
                                            r_x, r_y, r_z,
                                            g, rho_0, r_mass, r_r_mass, delta_t, r_delta_m, 
                                            r_A, r_C_d, r_c, x_frac, y_frac, z_frac)

                current_dist = np.sqrt((target_x[-1] - r_x)**2 + 
                                       (target_y[-1] - r_y)**2 + 
                                       (target_z[-1] - r_z)**2)
                time_projection = projection*(current_dist/radius)
                if(r_vz < 0):
                    r_vz = 0
                if(current_dist < 10):
                    r_vx = 0
                    r_vy = 0
                    r_vz = 0

                r_xs.append(r_x)
                r_ys.append(r_y)
                r_zs.append(r_z)
                r_vxs.append(r_vx)
                r_vys.append(r_vy)
                r_vzs.append(r_vz)

            else:
                proj_x.append(1000)
                proj_y.append(1000)
                proj_z.append(0)
                r_xs.append(1000)
                r_ys.append(1000)
                r_zs.append(0)
                r_vxs.append(0)
                r_vys.append(0)
                r_vzs.append(0)
        else:
            proj_x.append(1000)
            proj_y.append(1000)
            proj_z.append(0)
            r_xs.append(1000)
            r_ys.append(1000)
            r_zs.append(0)
            r_vxs.append(0)
            r_vys.append(0)
            r_vzs.append(0)


        # Hold on to current values
        xs.append(x)
        ys.append(y)
        zs.append(z)
        vxs.append(vx)
        vys.append(vy)
        vzs.append(vz)

        if(z < 1e-2 and i>100):
            # Match dimensions
            proj_x.append(proj_x[-1])
            proj_y.append(proj_y[-1])
            proj_z.append(proj_z[-1])
            inside.append(inside[-1])
            break

        try:
            if(current_dist <= 5):
                break
        except UnboundLocalError:
            pass

    end = True
    maxx = []
    maxy = []
    maxz = []
    fig, ax = plt.subplots()

    for i in range(len(xs)):
        maxx.append(np.max(xs[i]))
        maxy.append(np.max(ys[i]))
        maxz.append(np.max(zs[i]))

    kt = 1000
    phi, theta = np.mgrid[0.0:0.5*np.pi:180j, 0.0:2.0*np.pi:720j] # phi = alti, theta = azi
    dome_x = radius*np.sin(phi)*np.cos(theta) + center[0]
    dome_y = radius*np.sin(phi)*np.sin(theta) + center[1]
    dome_z = radius*np.cos(phi)
    for i in tqdm(range(0, len(xs))):
        plot(steps, i, xs[i], ys[i], zs[i],
             1.1*np.max(maxx), 1.1*np.max(maxy), 1.1*np.max(maxz), kt,
             center, radius, inside[i], proj_x[i], proj_y[i], proj_z[i], dome_x, dome_y, dome_z,
             r_xs[i], r_ys[i], r_zs[i])

    for i in tqdm(range(0, 20)):
        plot(steps, i+len(xs), xs[-1], ys[-1], zs[-1],
             1.1*np.max(maxx), 1.1*np.max(maxy), 1.1*np.max(maxz), kt,
             center, radius, inside[-1], proj_x[-1], proj_y[-1], proj_z[-1],
             dome_x, dome_y, dome_z,
             r_xs[-1], r_ys[-1], r_zs[-1], done=True)

    print("WRITING FILE")
    np.savetxt("./track_graph_missile/x_dat.txt", xs)
    np.savetxt("./track_graph_missile/y_dat.txt", ys)
    np.savetxt("./track_graph_missile/z_dat.txt", zs)
    np.savetxt("./track_graph_missile/x_vel.txt", vxs)
    np.savetxt("./track_graph_missile/y_vel.txt", vys)
    np.savetxt("./track_graph_missile/z_vel.txt", vzs)
    np.savetxt("./track_graph_missile/proj_x.txt", proj_x)
    np.savetxt("./track_graph_missile/proj_x.txt", proj_y)
    np.savetxt("./track_graph_missile/proj_x.txt", proj_z)
    np.savetxt("./track_graph_missile/r_x_dat.txt", r_xs)
    np.savetxt("./track_graph_missile/r_y_dat.txt", r_ys)
    np.savetxt("./track_graph_missile/r_z_dat.txt", r_zs)
    np.savetxt("./track_graph_missile/r_x_vel.txt", r_vxs)
    np.savetxt("./track_graph_missile/r_y_vel.txt", r_vys)
    np.savetxt("./track_graph_missile/r_z_vel.txt", r_vzs)
    fig, ax = plt.subplots()
    ax.plot(vxs, label="X VELOCITY")
    ax.plot(vys, label="Y VELOCITY")
    ax.plot(vzs, label="Z VELOCITY")

    ax.plot(estimation, evx, label="ESTIMATED X VELOCITY")
    ax.plot(estimation, evy, label="ESTIMATED Y VELOCITY")
    ax.plot(estimation, evz, label="ESTIMATED Z VELOCITY")

    ax.legend(loc='best')
    os.system("cd ./track_graph_missile/ ; ffmpeg -pattern_type glob -i \"[0-9]*.png\" -c:v libx264 -pix_fmt yuv420p -movflags +faststart output.mp4")
    plt.savefig("./track_graph_missile/velocities.png")


if __name__ == '__main__':
    rocket()

