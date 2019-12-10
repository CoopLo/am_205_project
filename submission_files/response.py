import numpy as np
import os
from numba import jit
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def num_approx_params(data, dt):
    # Numerically computes the position, velocity, acceleration, and jerk
    # Assuming the data is a length 4 vector of positions
    pos = data[3]
    vel = (data[3] - data[2])/dt
    acc = (data[3] - 2*data[2] + data[1])/dt**2
    temp_acc = (data[0] - 2*data[1] + data[2])/dt**2
    jerk = (acc - temp_acc)/dt
    return pos, vel, acc, jerk


def kinematics_acc(vx0, vy0, vz0, ax, ay, az, t, x0=0, y0=0, z0=0, g=0, vt=100):
    # from http://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node29.html
    '''
        vx{i} is input velocity
        ax{i} is input acceleration
        t  is time
        {i}0 is input position
        g is gravitational constant
        vt is terminal velocity

        NOTE: ONLY WAY TO CONTROL DRAG FORCE IS TO MODITY TERMINAL VELOCITY
    '''
    delta_t = 0.025
    x = x0 + vx0*vt/g*(1 - np.exp(-g*t/vt))*delta_t
    y = y0 + vy0*vt/g*(1 - np.exp(-g*t/vt))*delta_t
    z = z0 + (vt/g*(vz0 + vt)*(1 - np.exp(-g*t/vt)) - vt*t)*delta_t
    return x, y, z


#@jit
def rocket_equation(vx, vy, vz, t, x, y, z, g, rho_0, mass, r_mass, delta_t, delta_m, A, C_d, c,
                    vx_sched, vy_sched, vz_sched):
    '''
        v{i} is input velocity
        t is time
        {i} is input position
        g is gravity
        rho_0 is atmosphere density at sea level
        mass is non-fuel rocket mass
        r_mass is fuel mass
        delta_t is timestep
        delta_m is dm/dt (change in mass per unit time)
        A is cross-sectional area of rocket
        C_d is drag coefficient of rocket
        c is thrust coefficient
        v{i}_sched is the direction of velocity
    '''
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


def detect_project(center, radius, delta_t, g, num=0, detected=False):
    '''
        This function detects if the enemy rocket is within the detection radius.
        If in the radius, 4 data points are collected to calculate position, velocity,
        acceleration, and jerk. These are then used to project the enemy rockets position.
        The results are stored in /parallel_test/proj_{i}.txt for plotting and interception

        center is center of detection hemisphere
        radius is detection radius
        delta_t is timestep
        g is gravity
        num is projection number. Used to keep track of projections for data and plotting
    '''
    enemy_x, enemy_y, enemy_z = None, None, None
    ex, ey, ez = [], [], []

    detect = True
    start_len = 99999999999999
    while((len(ex) < 4 and len(ey) < 4 and len(ez) < 4) and not(detected)):
        try: # Make sure we have output to monitor

            # Read in data
            en_x = np.loadtxt("./parallel_test/enemy_x.txt")
            en_y = np.loadtxt("./parallel_test/enemy_y.txt")
            en_z = np.loadtxt("./parallel_test/enemy_z.txt")

            enemy_x = en_x[-1]
            enemy_y = en_y[-1]
            enemy_z = en_z[-1]

            # Check if enemy has been detected
            if(np.sqrt((enemy_x-center[0])**2 +
                       (enemy_y-center[1])**2 +
                       (enemy_z)**2) < radius and detect):

                start_len = len(en_x)
                detect = False

            # 4 data points have been measured since detection
            if(len(en_x) == start_len+4):
                print("\nSTART POINT: {}\n".format([enemy_x, enemy_y, enemy_z]))
                np.savetxt("./parallel_test/first_detection.txt", [int(len(en_x)-1)])
                ex = en_x[-4:]
                ey = en_y[-4:]
                ez = en_z[-4:]

        except (OSError, IndexError): # OSError catches no files, IndexError catches one line
            pass

    if(detected): # Missile has already been detected, only read last 4 measurements
        en_x = np.loadtxt("./parallel_test/enemy_x.txt")
        en_y = np.loadtxt("./parallel_test/enemy_y.txt")
        en_z = np.loadtxt("./parallel_test/enemy_z.txt")
        ex = en_x[-4:]
        ey = en_y[-4:]
        ez = en_z[-4:]

    # Approximate variables
    px, vx, ax, jx = num_approx_params(ex, delta_t)
    py, vy, ay, jy = num_approx_params(ey, delta_t)
    pz, vz, az, jz = num_approx_params(ez, delta_t)

    # project out 100 steps at a time?
    proj_x, proj_y, proj_z = [ex[-1]], [ey[-1]], [ez[-1]]
    for i in range(1, 200):
        # Kinematics for one step
        px, py, pz = kinematics_acc(vx, vy, vz, ax, ay, az, i*delta_t, px, py, pz, g)
        
        # Update velocity
        vx += ax*delta_t + 1/2*jx*delta_t**2
        vy += ay*delta_t + 1/2*jy*delta_t**2
        vz += az*delta_t + 1/2*jz*delta_t**2

        # Update acceleration
        ax += jx*delta_t
        ay += jy*delta_t
        az += jz*delta_t

        #print("UPDATED AFTER ONE STEP")
        #print(px, py, pz)
        proj_x.append(px)
        proj_y.append(py)
        proj_z.append(pz)
        if(pz < 0):
            break

    # Save projection data
    np.savetxt("./parallel_test/proj_x_{}.txt".format(num), proj_x)
    np.savetxt("./parallel_test/proj_y_{}.txt".format(num), proj_y)
    np.savetxt("./parallel_test/proj_z_{}.txt".format(num), proj_z)

    return num + 1, proj_x, proj_y, proj_z, True


def find_best_point(vx, vy, vz, x, y, z, g, rho_0, mass, r_mass, delta_t, delta_m, A, C_d,
                    c, px, py, pz):

    # Takes in all the same parameters as rocket_equation(*)
    # px, py, pz are projections
    ts = []
    for t_idx in range(0, len(px)):
        # Calculate thrust direction
        dist_vec = [(px[t_idx]-x), (py[t_idx]-y), (pz[t_idx]-z)]
        azimuth = np.arccos(dist_vec[2]/np.linalg.norm(dist_vec))
        radial = np.arctan(dist_vec[1]/dist_vec[0])

        # Set Thrust
        x_frac = dist_vec[0]/np.linalg.norm(dist_vec)
        y_frac = dist_vec[1]/np.linalg.norm(dist_vec)
        z_frac = dist_vec[2]/np.linalg.norm(dist_vec)

        # Temporary simulation variables
        t_vxs, t_vys, t_vzs = [0], [0], [0]
        xs, ys, zs = [x], [y], [z]

        # Need to compare with 
        last_dist = 2000
        #print("AIMING AT: {}".format(t_idx))
        for i in range(t_idx):
            tx, ty, tz, tvx, tvy, tvz, tr_mass = rocket_equation(
                                           t_vxs[-1], t_vys[-1], t_vzs[-1], i*delta_t,
                                           xs[-1], ys[-1], zs[-1], g, rho_0, mass, 
                                           r_mass, delta_t, delta_m, A, C_d, c,
                                           x_frac, y_frac, z_frac)
            
            xs.append(tx)
            ys.append(ty)
            zs.append(tz)
            t_vxs.append(vx)
            t_vys.append(vy)
            t_vzs.append(vz)

            # Distance between enemy and response
            dist = np.linalg.norm([tx - px[i], ty - py[i], tz - pz[i]])

            # Within hit radius
            if(dist < 2):
                print("HIT AT TIME: {}\n".format(i))
                ts.append((t_idx, i, dist, 'hit'))
                print("WE HIT IT RETURN NOW PLEASE")
                return [px[t_idx], py[t_idx], pz[t_idx]]

            # Getting farther away
            if(dist > last_dist):
                ts.append([t_idx, i-1, dist, 'away'])
                break

            if i >= t_idx-1:
                ts.append([t_idx, i, dist, 'last'])
            last_dist = dist

    #for i in ts:
    #    print(i)

    best_idx = np.argmin([i[3] for i in ts])
    #print(f'{best_idx}')
    #best_idx = ts[best_idx][1]
    best_point = [px[best_idx], py[best_idx], pz[best_idx]]
    #print("BEST AIM POINT: {}".format(best_point))
    #print(f'best point origin {x} {y} {z}')
    #print(f'rocket point {px[0]} {py[0]} {pz[0]}')
    #print(f'distance: {np.linalg.norm(np.array([x, y, z])) - np.linalg.norm(np.array([px[0], py[0], pz[0]]))}')

    return best_point


def response(delta_t=0.025):
    # Simulation variables
    steps = int(1000 * 0.025/delta_t)

    # Intercepting system variables
    center = [2000, 2000]
    radius = 1500
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
    c = 500000           # Exhaust force
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

    # Response missile
    projections = 0

    # Find ideal point
    collision = False
    #while(not(collision)):
    aim_points = []
    detected = False
    missed_height = False

    detect_time = 0

    for i in range(40):
        if missed_height is True:
            break

        # Project out
        projections, px, py, pz, detected = detect_project(center, radius, delta_t, g,
                                                           projections, detected)
        
        # Get best point
        best_point = find_best_point(vxs[-1], vys[-1], vzs[-1], xs[-1], ys[-1], zs[-1], g, rho_0,
                                     np.copy(mass), np.copy(r_mass), delta_t,
                                     delta_m, A, C_d, c, px, py, pz)

        # Hold on to best point
        if zs[-1] < best_point[2]:
            aim_points.append(best_point)

        # Calculate thrust direction
        dist_vec = [(best_point[0]-xs[-1]), 
                    (best_point[1]-ys[-1]), 
                    (best_point[2]-zs[-1])]
        azimuth = np.arccos(dist_vec[2]/np.linalg.norm(dist_vec))
        radial = np.arctan(dist_vec[1]/dist_vec[0])

        # Set Thrust
        x_frac = dist_vec[0]/np.linalg.norm(dist_vec)
        y_frac = dist_vec[1]/np.linalg.norm(dist_vec)
        z_frac = dist_vec[2]/np.linalg.norm(dist_vec)

        # Temporary simulation variables
        tx, ty, tz, tvx, tvy, tvz, tr_mass = 1000, 1000, 0, 0, 0, 0, np.copy(r_mass)
        t_vxs, t_vys, t_vzs = [0], [0], [0]

        # Run response misile in direction of best point
        for k in range(5):
            x, y, z, vx, vy, vz, r_mass = rocket_equation(
                                           vxs[-1], vys[-1], vzs[-1], delta_t,
                                           xs[-1], ys[-1], zs[-1], g, rho_0, mass,
                                           r_mass, delta_t, delta_m, A, C_d, c,
                                           x_frac, y_frac, z_frac)
            if(r_mass <= 0):
                break

            """if z > best_point[2]:
                missed_height = True
                break"""

            xs.append(x)
            ys.append(y)
            zs.append(z)
            vxs.append(vx)
            vys.append(vy)
            vzs.append(vz)

        # Sleep until we have enough new enemy data
        sleep_counter = 0
        while detect_time > len(np.loadtxt("./parallel_test/enemy_x.txt"))-5 and sleep_counter < 5:
            time.sleep(delta_t)
            sleep_counter += 1

        detect_time = len(np.loadtxt("./parallel_test/enemy_x.txt"))

    # Save response trajectory
    np.savetxt("./parallel_test/response_x.txt", xs)
    np.savetxt("./parallel_test/response_y.txt", ys)
    np.savetxt("./parallel_test/response_z.txt", zs)
    np.savetxt("./parallel_test/aim_points.txt", aim_points)


if __name__ == '__main__':
    rocket()

