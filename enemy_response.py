import numpy as np
from numba import jit
from enemy import rocket as enemy_rocket
from response import response as response_rocket
from multiprocessing import Process
from matplotlib import pyplot as plt
import os


def plot_enemy_projection():
    # Load enemy data
    ex = np.loadtxt("./parallel_test/enemy_x.txt")
    ey = np.loadtxt("./parallel_test/enemy_y.txt")
    ez = np.loadtxt("./parallel_test/enemy_z.txt")


    fig = plt.figure()
    print("PLOTTING ENEMY")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ex, ey, ez, label="ENEMY ROCKET PATH")

    # Get number of projections to plot
    dir_files = os.listdir("./parallel_test")
    projections = [s for s in dir_files if 'proj_x_' in s.lower()]
    max_proj = max([int(s.split('_')[2].split('.')[0]) for s in projections])

    # Plot projections
    for i in range(max_proj+1):
        px = np.loadtxt("./parallel_test/proj_x_{}.txt".format(i))
        py = np.loadtxt("./parallel_test/proj_y_{}.txt".format(i))
        pz = np.loadtxt("./parallel_test/proj_z_{}.txt".format(i))
        ax.plot(px, py, pz, label="ENEMY ROCKET PROJECTION {}".format(i))

    # Plot aim points
    print("PLOTTING AIM POINTS")
    aim_points = np.loadtxt("./parallel_test/aim_points.txt")
    for idx, apt in enumerate(aim_points):
        ax.scatter(*apt, marker='x', color='k', s=50, label="AIM POINT {}".format(idx))

    # Plot response path
    print("PLOTTING RESPONSE")
    r_x = np.loadtxt("./parallel_test/response_x.txt")
    r_y = np.loadtxt("./parallel_test/response_y.txt")
    r_z = np.loadtxt("./parallel_test/response_z.txt")
    ax.plot(r_x, r_y, r_z, color='r', label="RESPONSE ROCKET")

    # Check if successful interception
    first_detection = int(np.loadtxt("./parallel_test/first_detection.txt"))
    print("FIRST DETECTION TIME INDEX: {}".format(first_detection))
    dist_to_enemy = []
    print(ex[first_detection], ey[first_detection], ez[first_detection])
    for i in range(len(r_x)):
        r_vec = np.array([r_x[i], r_y[i], r_z[i]])
        e_vec = np.array([ex[i+first_detection], ey[i+first_detection], ez[i+first_detection]])
        dist_to_enemy.append(np.linalg.norm(r_vec - e_vec))
    print(dist_to_enemy)
    print(len(dist_to_enemy))
    print("CLOSEST DISTANCE: {}".format(min(dist_to_enemy)))

    # Plot detection hemisphere
    print("PLOTTING DOME")
    radius = 1500
    phi, theta = np.mgrid[0.0:0.5*np.pi:180j, 0.0:2.0*np.pi:720j]
    dome_x = radius*np.sin(phi)*np.cos(theta) + 2000
    dome_y = radius*np.sin(phi)*np.sin(theta) + 2000
    dome_z = radius*np.cos(phi)
    ax.plot_surface(dome_x, dome_y, dome_z, alpha=0.3, color='g')
    #ax.set(zlim=(0, max(max(ez), max(pz))))
    ax.set(zlim=(0, max(ez)))

    ax.legend(loc='best')
    plt.show()


def test_enemy_and_response():
    #delta_t = 0.025
    p1 = Process(target=enemy_rocket)
    p1.start()
    p2 = Process(target=response_rocket)
    p2.start()
    p1.join()
    p2.join()


if __name__ == '__main__':
    test_enemy_and_response()
    plot_enemy_projection()

